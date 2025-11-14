import argparse 

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import polars as pl

from .lib import parse_args, Data, Model, train_one_epoch, test, validate

def main(args : argparse.Namespace):
    
    df = pl.DataFrame()

    train_df, test_df = train_test_split(df, train_size=0.80, test_size=0.20)
    train_df, val_df  = train_test_split(train_df, train_size=0.90, test_size=0.10)

    train_dataset = Data(train_df)
    test_dataset = Data(test_df)
    val_dataset = Data(val_df)

    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)
    val_dataloader = DataLoader(val_dataset)

    opt = torch.optim.Adam(lr=args.learning_rate)
    criterion = torch.nn.L1Loss() # MAE loss

    model = Model()

    for e in range(args.epochs):
        train_one_epoch(train_dataloader, model, criterion, opt)

        validate(val_dataloader, model, criterion)

    
    test(test_dataloader, model)




if __name__ == "__main__":
    args = parse_args()
    main(args)