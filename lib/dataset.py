import torch
import torch.distributed as dist
import torch.nn.functional as F
import polars as pl
import numpy as np

from argparse import Namespace
from tqdm import tqdm
import requests
import os
import pickle

import transformers

from .config import PRETRAINED_BERT_MAX_TOKENS

class DataBaseClass(torch.utils.data.Dataset):
    def __init__(self, isSuper=False):
        """
        Initialize the dataset.
        """
        super(DataBaseClass, self).__init__()
        if not isSuper:
            NotImplementedError("Dont use this, use one of the children instead")


    def label_func(self, col_name: str, mapping: dict[str, int]) -> int:
        """
        Map labels to classes. Uses self.df attribute.

        Parameters
        ----------

        col_name : str
            The column name to label

        mapping : dict[str, int]
            The label -> Class ID mapping
        
        Returns
        -------
        The mapped labels as a PyTorch tensor.
        """

        mapping = {k.upper():v for k,v in mapping.items()}

        labels = self.df.select(pl.col(col_name).str.to_uppercase().replace(mapping)).to_numpy().astype(np.int32)

        labels = torch.from_numpy(labels.copy()).long() # .copy() because array is "not writable"?

        return labels

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
    
    def __len__(self) -> int:
        """
        Get the number of rows in the dataset.
        """
        return self.labels.shape[0]

class FitClassifierData(DataBaseClass):
    def __init__(self, df : pl.DataFrame, args : Namespace, name:str, tokenizer : transformers.AutoTokenizer = None):
        super(FitClassifierData, self).__init__(isSuper=True)
        
        self.tokenizer = tokenizer

        self.df = df
        self.args = args

        ## Give labels probability

        self.mapping = {'No Fit': 0, 'Potential Fit': 1, 'Good Fit': 2}
        self.labels = self.label_func("label", self.mapping)

        ## tokenize for BERT
        # check if cached first
        cache_file = os.path.join(args.dataset_dir, "fit", "cache", f"cached_{name}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                self.resumes, self.resumes_attention_mask, self.descriptions, self.descriptions_attention_mask = cached
            return
        else:
            cache_path = os.path.join(args.dataset_dir, "fit", "cache")
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

class TypeClassifierData(DataBaseClass):
    def __init__(self, df : pl.DataFrame, args : Namespace, name:str, tokenizer : transformers.AutoTokenizer = None):
        super(TypeClassifierData, self).__init__(isSuper=True)

        
        self.tokenizer = tokenizer

        self.df = df
        self.args = args

        ## Give labels probability

        self.mapping = {'Python Developer': 0, 'Designing': 1, 'Architecture': 2, 'Java Developer': 3, 'Database': 4, 'Aviation': 5, 'Business Analyst': 6, 'DevOps': 7, 'Digital Media': 8, 'PMO': 9, 'Civil Engineer': 10, 'ETL Developer': 11, 'Blockchain': 12, 'Data Science': 13, 'Agriculture': 14, 'Public Relations': 15, 'Apparel': 16, 'Arts': 17, 'Web Designing': 18, 'Accountant': 19, 'Mechanical Engineer': 20, 'React Developer': 21, 'Network Security Engineer': 22, 'Consultant': 23, 'SAP Developer': 24, 'Electrical Engineering': 25, 'Management': 26, 'Information Technology': 27, 'Health and Fitness': 28, 'Advocate': 29, 'Operations Manager': 30, 'Education': 31, 'Banking': 32, 'Automobile': 33, 'Finance': 34, 'Human Resources': 35, 'Sales': 36, 'Building and Construction': 37, 'BPO': 38, 'Testing': 39, 'DotNet Developer': 40, 'SQL Developer': 41, 'Food and Beverages': 42}
        self.labels = self.label_func("Category", self.mapping)

        ## tokenize for BERT
        # check if cached first
        cache_file = os.path.join(args.dataset_dir, "type", "cache", f"cached_{name}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                self.text, self.text_attention_mask = cached
            return
        else:
            cache_path = os.path.join(args.dataset_dir, "type", "cache")
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
        self.text, self.text_attention_mask = self.tokenize_and_chunk(self.tokenizer, self.df, "Text")

        # cache results
        cache = [self.text, self.text_attention_mask]

        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get individual rows from the dataset at a specified index.

        Parameters
        ----------

        i : int
            The index to access.

        Returns
        -------
        A **5-tuple** containing **(1)** the tokenized resume text, **(2)** the resume text's attention mask, 
        **(3)** the corresponding category.
        """

        return (
            self.text[i], self.text_attention_mask[i],
            self.labels[i]
        )
    

def download_dataset(args: Namespace) -> None:
    """
    Downloads dataset CSV files if `args.dataset_dir` does not exist.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed through the command line.
    """
    working_path = os.path.join(args.dataset_dir, "type" if args.is_type_classifier else "fit") # for the different datasets


    if os.path.exists(working_path):
        return
    
    os.makedirs(working_path, exist_ok=True)

    # TODO use https://www.openintro.org/data/index.php?data=resume if needing more data
    if args.is_type_classifier:
        DATASETS = {
            "full.csv": "https://huggingface.co/datasets/ahmedheakl/resume-atlas/resolve/main/train.csv?download=true",
            # only has one dataset.. we'll split it 80% train / 20% split
        }
    else:
        DATASETS = {
            "train.csv": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/train.csv?download=true",
            "test.csv": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/test.csv?download=true"
        }
    
    print("Dataset not downloaded. Starting download.")

    for file_name, url in DATASETS.items():
        file_path = os.path.join(working_path, file_name)

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

        if file_name == "full.csv":
            # split dataset
            print("Splitting downloaded dataset into 80% train and 20% test")

            df = pl.read_csv(file_path)

            df = df.sample(fraction=1.0, shuffle=True, seed=args.seed)

            train_size = int(df.shape[0] * 0.80)

            train_df = df.head(train_size)
            test_df = df.tail(-train_size)

            train_df.write_csv(os.path.join(working_path, "train.csv"))
            test_df.write_csv(os.path.join(working_path, "test.csv"))
        
class Word2VecData(torch.utils.data.Dataset):
    def __init__(self, df, args, w2v_vocab, max_len=300):
        self.args = args
        self.max_len = max_len
        self.word2idx = w2v_vocab # dict: word => index

        self.resumes = df["resume_text"].to_list()
        self.descriptions = df["job_description_text"].to_list()
        self.labels = df["label"].map_elements(self._label_func).to_numpy()

    def _label_func(self, s: str) -> int:
        s = s.upper()
        if "NO FIT" in s:
            return 0
        elif "POTENTIAL" in s:
            return 1
        elif "GOOD" in s:
            return 2
        else:
            raise ValueError(f"Unknown label: {s}")

    def _text_to_indices(self, text: str):
        tokens = text.lower().split()
        ids = [self.word2idx.get(tok, 0) for tok in tokens][:self.max_len]
        mask = [1] * len(ids)
        # pad
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, i):
        resume_text = self.resumes[i]
        desc_text = self.descriptions[i]

        resume_ids, resume_mask = self._text_to_indices(resume_text)
        desc_ids, desc_mask = self._text_to_indices(desc_text)
        label = torch.tensor(self.labels[i], dtype=torch.long)

        # same structure as Data class
        return resume_ids, resume_mask, desc_ids, desc_mask, label

    def __len__(self):
        return len(self.labels)
