import os
import torch
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from lib import parse_args, Word2VecLSTM, Word2VecData

from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(dataloader, model, device):
    all_true = []
    all_pred = []

    model.eval()
    for batch in dataloader:
        # batch structure for Word2VecData:
        # resume_ids, resume_mask, desc_ids, desc_mask, label
        resume_ids, resume_mask, desc_ids, desc_mask, labels = batch

        resume_ids = resume_ids.to(device)
        resume_mask = resume_mask.to(device)
        desc_ids = desc_ids.to(device)
        desc_mask = desc_mask.to(device)

        logits = model(resume_ids, resume_mask, desc_ids, desc_mask)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        all_pred.append(preds)
        all_true.append(labels)

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    return all_true, all_pred


def build_val_test_loaders(args, w2v_vocab):
    dataset_subdirectory = "type" if args.is_type_classifier else "fit"

    train_path = os.path.join(args.dataset_dir, dataset_subdirectory, "train.csv")
    test_path = os.path.join(args.dataset_dir, dataset_subdirectory, "test.csv")

    _ = pl.read_csv(train_path)  # optional
    test_df = pl.read_csv(test_path)

    # same val/test split as main.py
    test_df = test_df.sample(fraction=1, shuffle=True, seed=args.seed)
    split = int(test_df.shape[0] * 0.10)
    val_df = test_df.head(split)
    test_df = test_df.tail(-split)

    # use the vocab we got from the model
    val_dataset = Word2VecData(val_df, args, w2v_vocab)
    test_dataset = Word2VecData(test_df, args, w2v_vocab)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=1, pin_memory=True)

    return val_loader, test_loader



def plot_and_save_cm(y_true, y_pred, title, out_path):
    labels = [0, 1, 2]
    display_labels = ["No Fit", "Potential Fit", "Good Fit"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return cm


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make sure these match run used
    args.model_type = "Word2VecLSTM"
    args.version = "w2v_baseline" # subfolder name used under out/
    args.is_type_classifier = False # assuming fit classifier; set True if used type

    out_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)

    # load model checkpoint (choose best epoch)
    ckpt_path = os.path.join(out_dir, "model_epoch_7.pt")
    # construct model so it loads w2v.model and builds word2idx
    model = Word2VecLSTM(args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # handle both plain state_dict and dict-with-key
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    # build loaders using model's vocab
    val_loader, test_loader = build_val_test_loaders(args, model.word2idx)

    # validation
    y_val_true, y_val_pred = collect_predictions(val_loader, model, device)
    val_cm = plot_and_save_cm(
        y_val_true, y_val_pred,
        "Word2Vec LSTM – Validation Confusion Matrix",
        os.path.join(out_dir, "w2v_val_confusion.png"),
    )
    print("Validation classification report:")
    print(classification_report(y_val_true, y_val_pred, target_names=["No Fit", "Potential Fit", "Good Fit"]))

    # test
    y_test_true, y_test_pred = collect_predictions(test_loader, model, device)
    test_cm = plot_and_save_cm(
        y_test_true, y_test_pred,
        "Word2Vec LSTM – Test Confusion Matrix",
        os.path.join(out_dir, "w2v_test_confusion.png"),
    )
    print("Test classification report:")
    print(classification_report(y_test_true, y_test_pred, target_names=["No Fit", "Potential Fit", "Good Fit"]))

    print("Saved confusion matrices in:", out_dir)
    print("Val CM:\n", val_cm)
    print("Test CM:\n", test_cm)


if __name__ == "__main__":
    main()
