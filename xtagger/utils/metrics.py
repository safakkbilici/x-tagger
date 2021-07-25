import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score
)
import xtagger

def check_eval_metrics(metrics):
    if not metrics:
        raise ValueError("Metrics cannot be empty list.")
    else:
        for metric in metrics:
            if metric not in xtagger.IMPLEMENTED_METRICS:
                raise ValueError(f'"{metric}" not found in implemented metrics, current implemented metrics are: {xtagger.IMPLEMENTED_METRICS}')
        

def recall(y_true, y_pred):
    MODE = "avg_recall"
    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    recall_w = recall_score(y_true, y_pred, average="weighted")
    return {"Avg. Recall (w)": recall_w,
            "Avg. Recall (micro)": recall_micro,
            "Avg. Recall (macro)": recall_macro
    }

def precision(y_true, y_pred):
    MODE = "avg_precision"
    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    precision_w = precision_score(y_true, y_pred, average="weighted")
    return {"Avg. Precision (w)": precision_w,
            "Avg. Precision (micro)": precision_micro,
            "Avg. Precision (macro)": precision_macro
    }

def f1(y_true, y_pred):
    MODE = "avg_f1"
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    return {"Avg. F1 (w)": f1_w,
            "Avg. F1 (micro)": f1_micro,
            "Avg. F1 (macro)": f1_macro
    }

def accuracy(y_true, y_pred):
    MODE = "acc"
    acc = accuracy_score(y_true, y_pred)
    return acc

def tag2onehot(preds, ground_truth, tags):
    digit_pred = np.array([tags.index(tag) for tag in preds])
    digit_gt = np.array([tags.index(tag) for tag in ground_truth])

    preds_onehot = np.zeros((digit_pred.size, len(tags)))
    gt_onehot = np.zeros((digit_gt.size, len(tags)))

    preds_onehot[np.arange(digit_pred.size), digit_pred] = 1
    gt_onehot[np.arange(digit_gt.size), digit_gt] = 1

    return preds_onehot, gt_onehot
    
