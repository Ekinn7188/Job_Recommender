import torch
import polars as pl

from argparse import Namespace
from tqdm import tqdm
import requests
import os
import numpy as np 

import transformers

class Data(torch.utils.data.Dataset):
    def __init__(self, df : pl.DataFrame, args : Namespace):
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

        self.df = df
        self.args = args

        ## Give labels probability
        
        self.labels = df.select(pl.col("label").map_elements(self._label_func, return_dtype=pl.Float32)).to_numpy()
        self.labels = torch.from_numpy(self.labels.copy()) # .copy() because array is "not writable"?

        ## tokenize for BERT
        # Use pre-trained WordPiece tokenizer.. it'll break down the tokens better than one trained on our data.

        # https://huggingface.co//google-bert/bert-base-uncased
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

        # tokenize resumes...
        resumes_encoding : transformers.BatchEncoding = tokenizer(
            df.select(pl.col("resume_text")).to_numpy().flatten().tolist(), 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.args.max_tokens
        )

        self.resumes : torch.Tensor = resumes_encoding.input_ids
        self.resumes_attention_mask : torch.Tensor = resumes_encoding.attention_mask

        # tokenize descriptions...
        descriptions_encoding : transformers.BatchEncoding = tokenizer(
            df.select(pl.col("job_description_text")).to_numpy().flatten().tolist(), 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.args.max_tokens
        )

        self.descriptions : torch.Tensor = descriptions_encoding.input_ids
        self.descriptions_attention_mask : torch.Tensor = descriptions_encoding.attention_mask
    
    def _label_func(self, s: str) -> float:
        """
        Map labels to probabilities.

        Parameters
        ----------

        s : str
            The label to map. Must be one of `["No Fit", "Potential Fit", "Good Fit"]`

        Returns
        -------
        The mapped probability.
        """
        match s.upper():
            case "NO FIT":
                return 0.0
            case "POTENTIAL FIT":
                return self.args.potential_fit_probabiltiy
            case "GOOD FIT":
                return 1.0
            case _:
                raise ValueError(f'Expected "label" column to contain any of: ["No Fit", "Potential Fit", "Good Fit"]. Found: "{s}"')

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