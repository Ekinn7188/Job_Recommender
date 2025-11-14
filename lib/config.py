import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Job Recommender")

    # --------

    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--output_dir", default="./out")

    # --------

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--learning_rate", default=1e-4)

    # --------

    args = parser.parse_args()

    return args