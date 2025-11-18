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
    parser.add_argument("--version", default="model", type=str)

    # --------

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_type", default="SharedBERT", choices=["SharedBERT", "SplitBERT", "ML", "Word2Vec"])
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=float)
    parser.add_argument('--patience', type = int, default = 5, help = "# of epochs before early stopping")

    parser.add_argument("--potential_fit_probabiltiy", default=0.8, type=float)
    parser.add_argument("--max_tokens", default=7_680, type=int, help="For padding purposes. Must be a positive integer multiple of 512.") # 512 because of pretrained BERT's limitation


    # --------

    args = parser.parse_args()

    return args