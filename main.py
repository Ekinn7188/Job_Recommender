import argparse 
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import polars as pl
import numpy as np

from lib import parse_args, FitClassifierData, TypeClassifierData, TempModel, SharedBERT, \
    SplitBERT, train_one_epoch, test, validate, download_dataset, \
    TFIDFLogReg, Word2VecLSTM, Word2VecData, TypeClassifierBERT

def main(args : argparse.Namespace):
    # make output dirs
    os.makedirs(os.path.join(args.output_dir, args.version), exist_ok=True)

    # Get device

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # set up distributed data parallel
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank, rank, world_size, DEVICE = ddp_setup()

        if rank == 0:
            print(f"Running with {world_size} different GPUs on DDP")
    else:
        local_rank = 0
        rank = 0
        world_size = 1

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on {DEVICE}...")

    # Set random seeds for reproducibility
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    ## Read dataset CSVs

    dataset_subdirectory = "type" if args.is_type_classifier else "fit"

    train_path = os.path.join(args.dataset_dir, dataset_subdirectory, "train.csv")
    test_path = os.path.join(args.dataset_dir, dataset_subdirectory, "test.csv")

    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)

    ## Give random 10% of test.csv to validation
    # Random because dataset is sorted by label by default
    test_df = test_df.sample(fraction=1, shuffle=True, seed=args.seed)

    split = int(test_df.shape[0] * 0.10)
    val_df, test_df = test_df.head(split), test_df.tail(-split)
    
    # ML baseline doesn't use pytorch so special case
    if args.model_type.upper() == "ML":
        # train/test with TF-IDF + LogReg
        ml_model = TFIDFLogReg(args)
        val_metrics = ml_model.train()
        test_metrics = ml_model.test()

        # save metrics to output/<version>/
        out_dir = os.path.join(args.output_dir, args.version)
        os.makedirs(out_dir, exist_ok=True)

        val_df_out = pl.DataFrame(val_metrics)
        val_df_out.write_csv(os.path.join(out_dir, "ml_val_metrics.csv"))

        test_df_out = pl.DataFrame(test_metrics)
        test_df_out.write_csv(os.path.join(out_dir, "ml_test_metrics.csv"))

        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)
        return

    # Prepare data
    if rank == 0:
        print("Preparing datasets...\n")

    if args.model_type.upper() == "WORD2VEC":
        # load vocab from W2V model (only need vocab, not full model)
        tmp_model = Word2VecLSTM(args)
        w2v_vocab = tmp_model.word2idx

        train_dataset = Word2VecData(train_df, args, w2v_vocab)
        test_dataset  = Word2VecData(test_df,  args, w2v_vocab)
        val_dataset   = Word2VecData(val_df,   args, w2v_vocab)

    else:
        if args.is_type_classifier:
            data_class = TypeClassifierData
        else:
            data_class = FitClassifierData
            
        train_dataset = data_class(train_df, args, "train")
        test_dataset  = data_class(test_df, args, "test", tokenizer=train_dataset.tokenizer)
        val_dataset   = data_class(val_df, args, "val", tokenizer=train_dataset.tokenizer)

    if rank == 0:
        print("Finished datasets\n")

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size//world_size, sampler=train_sampler, num_workers=1, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size//world_size, sampler=test_sampler, num_workers=1, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size//world_size, sampler=val_sampler, num_workers=1, pin_memory=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=False)


    # Get model

    match (args.model_type.upper()):
        case "SHAREDBERT":
            model = SharedBERT(args)
        case "SPLITBERT":
            model = SplitBERT(args)
        case "ML":
            raise ValueError("ML model_type uses separate TF-IDF code path in main().")
        case "WORD2VEC":
            model = Word2VecLSTM(args)
        case "TYPECLASSIFIERBERT":
            model = TypeClassifierBERT(args)
        case _:
            raise ValueError(f'Invalid model type selected. Must be one of: ["SharedBERT", "SplitBERT", "ML", "Word2Vec"]. Currently selected: {args.model_type}.')
    
    model = model.to(DEVICE)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Smaller LR on BERT, bigger on head.

    if "BERT" in args.model_type.upper(): 
        bert_params = []
        non_bert_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Any parameter belonging to either BERT encoder
            if "BERT_encoder" in name:
                bert_params.append(param)
            else:
                non_bert_params.append(param)
        
        opt = torch.optim.Adam([
            {"params": bert_params, "lr": 5e-5},
            {"params": non_bert_params, "lr": args.learning_rate},
        ])
    else:
        opt = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
        
    criterion = torch.nn.CrossEntropyLoss()

    records = []
    patience = args.patience
    best_val_loss = float("inf")

    if rank == 0:
        print(f"Starting training...\n")

    # Start training
    for e in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(e)

        training_CE, training_accuracy, training_precision, training_recall  = train_one_epoch(train_dataloader, model, criterion, opt, DEVICE, args)

        val_CE, val_accuracy, val_precision, val_recall  = validate(val_dataloader, model, DEVICE, args)
 
        if rank == 0:        
            records.append({
                "epoch": e+1,
                "training_CE": training_CE,
                "training_accuracy": training_accuracy,
                "training_precision": training_precision,
                "training_recall": training_recall,
                "validation_CE": val_CE,
                "validation_accuracy": val_accuracy,
                "validation_precision": val_precision,
                "validation_recall": val_recall,
            })

            model_path = os.path.join(args.output_dir, args.version, f"model_epoch_{e+1}.pt")
            if is_ddp:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)


            print(
                f"Epoch [{e + 1:02d}/{args.epochs}] "
                f"Train CE Loss: {training_CE:.4f} | Train Accuracy {training_accuracy:.4f} | "
                f"Val CE Loss: {val_CE:.4f} | Val Accuracy {val_accuracy:.4f}"
            )

            if best_val_loss > val_CE:
                best_val_loss = val_CE
                patience = args.patience
                early_stop_flag = False
            else:
                patience -= 1
                if patience < 0:
                    early_stop_flag = True
                else:
                    early_stop_flag = False
        else:
            early_stop_flag = False
        

        if is_ddp:
            # Give all ranks the early_stop_flag from rank 0
            flag_tensor = torch.tensor([early_stop_flag], device=DEVICE)
            torch.distributed.broadcast(flag_tensor, src=0)
            early_stop_flag = int(flag_tensor.item())

        if early_stop_flag:
            if rank == 0:
                print("Stopping fold early...")
            break


    # Save per-fold logs

    if rank == 0:
        log_df = pl.DataFrame(records)
        log_path = os.path.join(args.output_dir, args.version, f"output_log.csv")
        log_df.write_csv(log_path)

    test_CE, test_accuracy, test_precision, test_recall = test(test_dataloader, model, DEVICE, args)

    if rank == 0:    
        test_results = pl.DataFrame({
            "test_CE": test_CE,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
        })

        log_path = os.path.join(args.output_dir, args.version, f"test_results.csv")
        test_results.write_csv(log_path)

    # cleanup process
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    return local_rank, rank, world_size, device


if __name__ == "__main__":
    args = parse_args()
    download_dataset(args)
    main(args)