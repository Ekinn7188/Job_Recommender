import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from lib import parse_args, TFIDFLogReg


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
    args.model_type = "ML"
    args.version = "tfidf_baseline" # folder under out/ that was used

    out_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)

    ml_model = TFIDFLogReg(args)

    # train once so vectorizer + model are fit
    val_metrics = ml_model.train()
    print("Val metrics from train():", val_metrics)

    # get validation predictions (they come from 10% split inside train())
    X_train, y_train = ml_model.load_split("train")
    indices = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in indices]
    y_train = y_train[indices]
    n_train = int(0.9 * len(X_train))

    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    X_val = ml_model.vectorizer.transform(X_val)
    y_val_pred = ml_model.model.predict(X_val)

    val_cm = plot_and_save_cm(
        y_val, y_val_pred,
        "TF-IDF + LogReg – Validation Confusion Matrix",
        os.path.join(out_dir, "tfidf_val_confusion.png"),
    )
    print("Validation report:")
    print(classification_report(y_val, y_val_pred, target_names=["No Fit", "Potential Fit", "Good Fit"]))

    # test predictions using the class's test() logic pattern but keeping preds
    X_test, y_test = ml_model.load_split("test")
    X_test_t = ml_model.vectorizer.transform(X_test)
    y_test_pred = ml_model.model.predict(X_test_t)

    test_cm = plot_and_save_cm(
        y_test, y_test_pred,
        "TF-IDF + LogReg – Test Confusion Matrix",
        os.path.join(out_dir, "tfidf_test_confusion.png"),
    )
    print("Test report:")
    print(classification_report(y_test, y_test_pred, target_names=["No Fit", "Potential Fit", "Good Fit"]))

    print("Saved TF-IDF confusion matrices to:", out_dir)
    print("Val CM:\n", val_cm)
    print("Test CM:\n", test_cm)


if __name__ == "__main__":
    main()
