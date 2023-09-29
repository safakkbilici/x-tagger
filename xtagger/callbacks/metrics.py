from typing import List, Union, Dict

import numpy as np
from xtagger.callbacks.metrics_ import *


def convert_to_onehot(preds: List[str], ground_truth: List[str], tags: List[str]):
    digit_pred = np.array([tags.index(tag) for tag in preds])
    digit_gt = np.array([tags.index(tag) for tag in ground_truth])

    preds_onehot = np.zeros((digit_pred.size, len(tags)))
    gt_onehot = np.zeros((digit_gt.size, len(tags)))

    preds_onehot[np.arange(digit_pred.size), digit_pred] = 1
    gt_onehot[np.arange(digit_gt.size), digit_gt] = 1

    return preds_onehot, gt_onehot


def metric_results(
    y_true: Union[np.ndarray, List[List[int]]],
    y_pred: Union[np.ndarray, List[List[int]]],
    eval_metrics: List[BaseMetric],
    tags: List[str],
) -> Dict[str, Dict[str, float]]:
    results = {}

    for metric in eval_metrics:
        results[metric.mode] = metric(y_true, y_pred, tags)()
    return results
