from typing import List, Union, Dict

import numpy as np
from xtagger.callbacks.metrics_ import *


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
) -> Dict[str, float | Dict[float]]:
    results = {}

    for metric in eval_metrics:
        results[metric.mode] = metric(gt, preds, tags)()
    return results
