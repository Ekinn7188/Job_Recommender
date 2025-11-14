import torch
import polars as pl

from argparse import Namespace
from tqdm import tqdm
import requests
import os
import time

class Data(torch.utils.data.Dataset):
    def __init__(self, df : pl.DataFrame):
        super(Data, self).__init__()

        self.df = df

        # TODO do some processing
    
    def __getitem__(self, i: int):
        # TODO retrieve data
        return torch.tensor([0])
    
    def __len__(self):
        # TODO return dataset length
        return 0

def download_dataset(args: Namespace):
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

    exit(0)