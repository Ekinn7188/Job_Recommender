import argparse 
import os

import torch
from torch.utils.data import DataLoader

import polars as pl
import numpy as np

from lib import parse_args, Data, TempModel, SharedBERT, train_one_epoch, test, validate, download_dataset

def main(args : argparse.Namespace):
    # make output dirs
    os.makedirs(os.path.join(args.output_dir, args.version), exist_ok=True)

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
    test_dataset = Data(test_df, args, tokenizer=train_dataset.tokenizer)
    print("Finished test")
    val_dataset = Data(val_df, args, tokenizer=train_dataset.tokenizer)
    print("Finished val\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Get device

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")
    

    # Get model

    match (args.model_type.upper()):
        case "SHAREDBERT":
            model = SharedBERT(args)
        case "SPLITBERT":
            model = TempModel()
        case "ML":
            model = TempModel()
        case "WORD2VEC":
            model = TempModel()
        case _:
            raise ValueError(f'Invalid model type selected. Must be one of: ["SharedBERT", "SplitBERT", "ML", "Word2Vec"]. Currently selected: {args.model_type}.')
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    opt = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss() # MAE loss


    records = []
    patience = args.patience
    best_val_loss = float("inf")

    print(f"Starting training...\n")

    # Start training
    for e in range(args.epochs):
        training_MAE, training_spearman_coeff, training_pearson_coeff  = train_one_epoch(train_dataloader, model, criterion, opt, DEVICE)

        val_MAE, val_spearman_coeff, val_pearson_coeff  = validate(val_dataloader, model, DEVICE)

        records.append({
            "epoch": e+1,
            "train_MAE": training_MAE,
            "training_spearman": training_spearman_coeff,
            "training_pearson": training_pearson_coeff,
            "validation_MAE": val_MAE,
            "validation_spearman": val_spearman_coeff,
            "validation_pearson": val_pearson_coeff,
        })


        model_path = os.path.join(args.output_dir, args.version, f"model_epoch_{e+1}.csv")

        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)


        print(
            f"Epoch [{e + 1:02d}/{args.epochs}] "
            f"Train MAE Loss: {training_MAE:.4f} | Train Spearman Corr. {training_spearman_coeff:.4f} | Train Pearson Corr. {training_pearson_coeff:.4f}"
            f"Val Loss: {val_MAE:.4f} | Val Spearman Corr. {val_spearman_coeff:.4f} | Val Pearson Corr. {val_pearson_coeff:.4f}"
        )

        if best_val_loss > val_MAE:
            best_val_loss = val_MAE
            patience = args.patience
        else:
            patience -= 1
            if patience < 0:
                print("Stopping fold early...")
                break

    # Save per-fold logs
    log_df = pl.DataFrame(records)
    log_path = os.path.join(args.output_dir, args.version, f"output_log.csv")
    log_df.write_csv(log_path)

    test_MAE, test_spearman_coeff, test_pearson_coeff  = test(test_dataloader, model, DEVICE)

    
    test_results = pl.DataFrame({
        "test_MAE": test_MAE,
        "test_spearman": test_spearman_coeff,
        "test_pearson": test_pearson_coeff,
    })
    log_path = os.path.join(args.output_dir, args.version, f"test_results.csv")
    test_results.write_csv(log_path)




if __name__ == "__main__":
    args = parse_args()
    download_dataset(args)
    main(args)