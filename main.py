import argparse 
import os

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import polars as pl

from lib import parse_args, Data, Model, train_one_epoch, test, validate, download_dataset

def main(args : argparse.Namespace):
    ## Read dataset CSVs

    train_path = os.path.join(args.dataset_dir, "train.csv")
    test_path = os.path.join(args.dataset_dir, "test.csv")

    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)

    ## Give random 10% of test.csv to validation
    # Random because dataset is sorted by label by default
    test_df = test_df.sample(fraction=1, shuffle=True, seed=0) # seed=0 for reproducibility

    split = int(test_df.shape[0] * 0.10)
    val_df, test_df = test_df.head(split), test_df.tail(-split)

    # Prepare data


    # val_dataset = Data(val_df)

    train_dataset = Data(train_df, args)
    test_dataset = Data(test_df, args)
    val_dataset = Data(val_df, args)

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
    download_dataset(args)
    main(args)