import torch
import torch.distributed as dist
import torch.nn.functional as F
import polars as pl

from argparse import Namespace
from tqdm import tqdm
import requests
import os
import pickle

import transformers

from .config import PRETRAINED_BERT_MAX_TOKENS

class Data(torch.utils.data.Dataset):
    def __init__(self, df : pl.DataFrame, args : Namespace, name:str, tokenizer : transformers.AutoTokenizer = None):
        """
        Initialize the dataset.

        Parameters
        ----------

        df : polars.DataFrame
            The dataframe to process. Must contain columns `["resume_text", "job_description_text", "label"]`. </br>
            The column named `label` must contain rows holding `["No Fit", "Potential Fit", "Good Fit"]`.
        args : argparse.Namespace
            The arguments passed through the command line.
        """
        super(Data, self).__init__()

        self.tokenizer = tokenizer

        self.df = df
        self.args = args

        ## Give labels probability
        
        self.labels = self.df.select(pl.col("label").map_elements(self._label_func, return_dtype=pl.Int32)).to_numpy()
        self.labels = torch.from_numpy(self.labels.copy()).long() # .copy() because array is "not writable"?

        ## tokenize for BERT
        # check if cached first
        cache_file = os.path.join(args.dataset_dir, "cache", f"cached_{name}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                self.resumes, self.resumes_attention_mask, self.descriptions, self.descriptions_attention_mask = cached
            return
        else:
            cache_path = os.path.join(args.dataset_dir, "cache")
            os.makedirs(cache_path, exist_ok=True)

        # Use pre-trained WordPiece tokenizer.. it'll break down the tokens better than one trained on our data.

        # https://huggingface.co//google-bert/bert-base-uncased
        if not self.tokenizer:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

            if rank == 0:
                print("loading pretrained BERT tokenizer...")

            self.tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")

            if rank == 0:
                print("loaded pretrained BERT tokenizer.\n")

        # check that max_tokens is a multiple of PRETRAINED_BERT_MAX_TOKENS (512)
        assert self.args.max_tokens % PRETRAINED_BERT_MAX_TOKENS == 0 and self.args.max_tokens > 0, "The configurated max_tokens value must be a multiple of 512, which is greater than 0."

        # tokenize resumes...
        self.resumes, self.resumes_attention_mask = self.tokenize_and_chunk(self.tokenizer, self.df, "resume_text")

        # tokenize descriptions...
        self.descriptions, self.descriptions_attention_mask = self.tokenize_and_chunk(self.tokenizer, self.df, "job_description_text")
        
        # cache results
        cache = [self.resumes, self.resumes_attention_mask, self.descriptions, self.descriptions_attention_mask]
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)


    def _label_func(self, s: str) -> int:
        """
        Map labels to classes.

        Parameters
        ----------

        s : str
            The label to map. Must be one of `["No Fit", "Potential Fit", "Good Fit"]`

        Returns
        -------
        The mapped class.
        """
        match s.upper():
            case "NO FIT":
                return 0
            case "POTENTIAL FIT":
                return 1
            case "GOOD FIT":
                return 2
            case _:
                raise ValueError(f'Expected "label" column to contain any of: ["No Fit", "Potential Fit", "Good Fit"]. Found: "{s}"')

    def tokenize_and_chunk(self, tokenizer: transformers.BertTokenizerFast, df: pl.DataFrame, col: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the strings and split them into chunks of PRETRAINED_BERT_MAX_TOKENS (512). The pre-trained BERT model can only handle 512 tokens at a time.

        Parameters
        ----------

        tokenizer : transformers.BertTokenizerFast
            The pre-trained BERT tokenizer.
        
        df : pl.DataFrame
            The dataset that the strings are coming from

        col : str
            The column in the dataset that the strings are located in

        Returns
        -------

        A **2-tuple** containing **(1)** the encoded token sequence and **(2)** the attention mask for the sequence.

        """

        if dist.is_initialized():
            os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid deadlocks with the tokenizer

        docs = df.select(pl.col(col)).to_numpy().flatten().tolist()

        encoding : transformers.BatchEncoding = tokenizer(
            docs, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=PRETRAINED_BERT_MAX_TOKENS,
            return_overflowing_tokens=True,
            stride=0,
        )
        
        id : torch.Tensor = encoding.input_ids
        attention_mask : torch.Tensor = encoding.attention_mask
        overflow_mapping : torch.Tensor = encoding.overflow_to_sample_mapping

        # map overflow to correct documents

        num_docs = len(docs)
        ids = [[] for _ in range(num_docs)]
        masks = [[] for _ in range(num_docs)]

        for chunk_idx, doc_idx in enumerate(overflow_mapping):
            i = int(doc_idx.item())
            ids[i].append(id[chunk_idx])
            masks[i].append(attention_mask[chunk_idx])

        # join to one tensor

        ids = [torch.stack(chunks) for chunks in ids]     
        masks = [torch.stack(chunks) for chunks in masks]

        max_chunks = self.args.max_tokens // PRETRAINED_BERT_MAX_TOKENS

        ids = [
            F.pad(chunks, pad=(0,0,0, max_chunks-len(chunks))) if len(chunks) < max_chunks
            else chunks[:max_chunks]
            for chunks in ids 
        ]
        masks = [
            F.pad(chunks, pad=(0,0,0, max_chunks-len(chunks))) if len(chunks) < max_chunks
            else chunks[:max_chunks]
            for chunks in masks 
        ]

        ids = torch.stack(ids)
        masks = torch.stack(masks)

        return ids, masks


    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get individual rows from the dataset at a specified index.

        Parameters
        ----------

        i : int
            The index to access.

        Returns
        -------
        A **5-tuple** containing **(1)** the tokenized resume text, **(2)** the resume text's attention mask, 
        **(3)** the tokenized description text, **(4)** the description text's attention mask,
        and **(5)** the corresponding fit probability.
        """

        return (
            self.resumes[i], self.resumes_attention_mask[i],
            self.descriptions[i], self.descriptions_attention_mask[i],
            self.labels[i]
        )
    
    def __len__(self) -> int:
        """
        Get the number of rows in the dataset.
        """
        return self.labels.shape[0]

def download_dataset(args: Namespace) -> None:
    """
    Downloads dataset CSV files if `args.dataset_dir` does not exist.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed through the command line.
    """

    if os.path.exists(args.dataset_dir):
        return
    
    os.makedirs(args.dataset_dir, exist_ok=True)

    # TODO use https://www.openintro.org/data/index.php?data=resume if needing more data
    DATASETS = {
        "train.csv": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/train.csv?download=true",
        "test.csv": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/test.csv?download=true"
    }
    
    print("Dataset not downloaded. Starting download.")

    for file_name, url in DATASETS.items():
        file_path = os.path.join(args.dataset_dir, file_name)

        r = requests.get(url, stream=True)
        
        file_size = int(r.headers.get('content-length', 0))

        chunk_size = 1_024 # 1 KB

        num_chunks = file_size // chunk_size

        bytes_downloaded = 0

        progress_bar = tqdm(
            total=file_size, 
            desc=f"Downloading {file_name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1_024, 
            ncols=175,
            ascii=" >=",
            leave=True
        )

        with open(file_path, "wb") as file:
            for i, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
                file.write(chunk)

                if i != num_chunks-1:
                    bytes_downloaded += chunk_size
                    progress_bar.update(chunk_size)
        
        progress_bar.close()