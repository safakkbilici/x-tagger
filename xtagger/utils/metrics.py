import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score
)

def f1(y_true, y_pred):
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    return {"Avg. F1 (w)": f1_w, "Avg. F1 (micro)": f1_micro, "Avg. F1 (macro)": f1_macro}

def accuracy(y_true, y_pred):
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
    
