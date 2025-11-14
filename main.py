import argparse 
import os

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import polars as pl
import numpy as np

from lib import parse_args, Data, TempModel, SharedBERT, train_one_epoch, test, validate, download_dataset

def main(args : argparse.Namespace):
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## Read dataset CSVs

    train_path = os.path.join(args.dataset_dir, "train.csv")
    test_path = os.path.join(args.dataset_dir, "test.csv")

    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)

    ## Give random 10% of test.csv to validation
    # Random because dataset is sorted by label by default
    test_df = test_df.sample(fraction=1, shuffle=True, seed=args.seed)

    split = int(test_df.shape[0] * 0.10)
    val_df, test_df = test_df.head(split), test_df.tail(-split)

    # Prepare data

    print("Preparing datasets...\n")

    train_dataset = Data(train_df, args)
    print("Finished train")
    test_dataset = Data(test_df, args)
    print("Finished test")
    val_dataset = Data(val_df, args)
    print("Finished val\n")

    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)
    val_dataloader = DataLoader(val_dataset)

    # Get device

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")
    

    # Get model

    match (args.model_type.upper()):
        case "SHAREDBERT":
            model = SharedBERT(args).to(DEVICE)
        case "SPLITBERT":
            model = TempModel().to(DEVICE)
        case "ML":
            model = TempModel().to(DEVICE)
        case "WORD2VEC":
            model = TempModel().to(DEVICE)
        case _:
            raise ValueError(f'Invalid model type selected. Must be one of: ["SharedBERT", "SplitBERT", "ML", "WORD2VEC"]. Currently selected: {args.model_type}.')
    
    opt = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss() # MAE loss


    print(f"Starting training...\n")

    # Start training
    for e in range(args.epochs):
        train_one_epoch(train_dataloader, model, criterion, opt, DEVICE)

        validate(val_dataloader, model, criterion, DEVICE)

    test(test_dataloader, model, DEVICE)




if __name__ == "__main__":
    args = parse_args()
    download_dataset(args)
    main(args)