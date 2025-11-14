import torch
import torch.utils.data as Dataset 

import polars as pl

class Data(Dataset):
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
