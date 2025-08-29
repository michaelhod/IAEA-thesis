import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef, log_loss, roc_auc_score,
    top_k_accuracy_score
)
def metrics(y_pred, y_true):
    # 1) Ground truth: your string with ? = 3
    # y_true_str = "2 2 1 1 ? ? ? ? 3 3 1 1 3 3 2 2 2 2 1 1 ? ? 1 1 3 3 1 1 3 3 1 1 1 1 2 2 2 2 2 2"
    # y_true = [3 if tok == "?" else int(tok) for tok in y_true_str.split()]
    classes = [1, 2, 3]

    # 2) Predictions from your model (fill these in; length must match y_true)
    # Example placeholder (replace with your model's predicted labels):
    # y_pred = [ ... same length as y_true, values in {1,2,3} ... ]

    # ---- Basic label-based metrics ----
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # print("Samples:", len(y_true_np))
    # print("Classes:", classes)
    # print()

    # Accuracy & Balanced Accuracy
    print("Accuracy:", accuracy_score(y_true_np, y_pred_np))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true_np, y_pred_np))
    print()

    # Confusion matrices
    cm = confusion_matrix(y_true_np, y_pred_np, labels=classes)
    cm_norm = confusion_matrix(y_true_np, y_pred_np, labels=classes, normalize="true")
    # print("Confusion Matrix (rows=true, cols=pred):\n", cm)
    # print("\nConfusion Matrix (normalized by true class):\n", np.round(cm_norm, 3))
    # print()

    # Per-class precision/recall/F1 + macro/micro/weighted
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=classes, zero_division=0
    )
    # print("Per-class metrics (order: class 1, 2, 3)")
    # for i, c in enumerate(classes):
    #     print(f" Class {c}: precision={prec[i]:.4f}  recall={rec[i]:.4f}  f1={f1[i]:.4f}  support={support[i]}")
    # print()

    #print("Averaged metrics:")
    for avg in ["macro", "micro", "weighted"]:
        p, r, f, s = precision_recall_fscore_support(
            y_true_np, y_pred_np, average=avg, zero_division=0
        )
    #     print(f" {avg.capitalize():7s} -> precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")
    # print()

    # Cohen's kappa & Matthews correlation coefficient
    # print("Cohen's kappa:", cohen_kappa_score(y_true_np, y_pred_np))
    # print("Matthews corrcoef (MCC):", matthews_corrcoef(y_true_np, y_pred_np))
    # print()

    # Sklearn's formatted report (nice summary)
    print("Classification report:\n")
    print(classification_report(y_true_np, y_pred_np, labels=classes, digits=4, zero_division=0))