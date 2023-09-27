from typing import List, Union

import numpy as np
from xtagger.callbacks.metrics_ import *

METRICS = {
    "f1": F1,
    "precision": Precision,
    "recall": Recall,
    "accuracy": Accuracy,
    "multiclass_precision": MultiClassPrecision,
    "multiclass_f1": MultiClassF1,
    "multiclass_recall": MultiClassRecall,
}


def tag2onehot(preds: List[str], ground_truth: List[str], tags: List[str]):
    digit_pred = np.array([tags.index(tag) for tag in preds])
    digit_gt = np.array([tags.index(tag) for tag in ground_truth])

    preds_onehot = np.zeros((digit_pred.size, len(tags)))
    gt_onehot = np.zeros((digit_gt.size, len(tags)))

    preds_onehot[np.arange(digit_pred.size), digit_pred] = 1
    gt_onehot[np.arange(digit_gt.size), digit_gt] = 1

    return preds_onehot, gt_onehot


def metric_results(
    gt: Union[np.ndarray, List[List[int]]],
    preds: Union[np.ndarray, List[List[int]]],
    eval_metrics: List[str],
    tags: List[str],
) -> dict:
    results = {}

    for metric in eval_metrics:
        if metric in METRICS:
            results[metric] = METRICS[metric](gt, preds, tags)()

        elif metric == "report":
            print(classification_report(gt, preds, target_names=tags))

        elif type(metric) != str and metric.__bases__[0] == BaseMetric:
            user_metric = metric(gt, preds, tags)
            results[str(metric.__name__)] = user_metric()
    return results
