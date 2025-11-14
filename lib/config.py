import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse all the command line arguments. None are required. Defaults are set here.
    """

    parser = argparse.ArgumentParser("Job Recommender")

    # --------

    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--output_dir", default="./out", type=str)

    # --------

    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--potential_fit_probabiltiy", default=0.8, type=float)
    parser.add_argument("--max_tokens", default=7_500, type=int)

    # --------

    args = parser.parse_args()

    return args