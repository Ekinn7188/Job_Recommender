import argparse

# Decided by Google when they trained BERT.
PRETRAINED_BERT_MAX_TOKENS = 512

def parse_args() -> argparse.Namespace:
    """
    Parse all the command line arguments. None are required. Defaults are set here.
    """

    parser = argparse.ArgumentParser("Job Recommender")

    # --------

    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--output_dir", default="./out", type=str)
    parser.add_argument("--models_dir", default="./models", type=str)
    parser.add_argument("--version", default="model", type=str)

    # --------

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--model_type", default="SharedBERT", choices=["FitClassifierBERT", "TypeClassifierBERT", "ML", "Word2Vec"])
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--patience', type = int, default = 5, help = "# of epochs before early stopping")

    parser.add_argument("--max_tokens", default=7_680, type=int, help="For padding purposes. Must be a positive integer multiple of 512.") # 512 because of pretrained BERT's limitation


    # --------

    args = parser.parse_args()

    # add a type classifier vs fit classifier argument for simplicity
    if "TypeClassifier" in args.model_type:
        type_classifier = True
    else:
        type_classifier = False
    
    setattr(args, "is_type_classifier", type_classifier) # can now be accessed via args.is_type_classifier

    return args