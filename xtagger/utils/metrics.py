import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    classification_report
)
import xtagger
from typing import List, Optional, Tuple, Union

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning) 

class xMetrics():
    def __init__(self, y_true, y_pred, tags=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tags = tags

def check_eval_metrics(metrics):
    if not metrics:
        raise ValueError("Metrics cannot be empty list.")
    else:
        for metric in metrics:
            if metric not in xtagger.IMPLEMENTED_METRICS:
                if type(metric) == str:
                    raise ValueError(f'"{metric}" not found in implemented metrics, current implemented metrics are: {xtagger.IMPLEMENTED_METRICS}')
            elif type(metric) != str and metric.__bases__[0] != xMetrics:
                raise ValueError(f'User defined metric {metric} must be inherited from xtagger.utils.metrics.xMetrics')


def metric_results(
        gt: Union[np.ndarray, List[List[int]]],
        preds: Union[np.ndarray, List[List[int]]],
        eval_metrics: List[str],
        result_type: str,
        tags: List[str]
) -> dict:
    results = {}
    result_t = 100 if result_type=="%" else 1

    for metric_ in eval_metrics:
    
        if metric_ == "avg_f1":
            f1s = f1(gt, preds)
            f1s.update((key, value * result_t) for key, value in f1s.items())
            results["avg_f1"] = f1s
            
        if metric_ == "acc":
            acc = accuracy(gt, preds) * result_t
            results["acc"] = acc
        
        if metric_ == "avg_recall":
            recalls = recall(gt, preds)
            recalls.update((key, value * result_t) for key, value in recalls.items())
            results["avg_recall"] = recalls

        if metric_ == "avg_precision":
            precisions = precision(gt, preds)
            precisions.update((key, value * result_t) for key, value in precisions.items())
            results["avg_precision"] = precisions

        if metric_ == "classwise_f1":
            c_f1s = multiclass_f1(gt, preds, tags)
            c_f1s.update((key, value * result_t) for key, value in c_f1s.items())
            results["classwise_f1"] = c_f1s

        if metric_ == "classwise_precision":
            c_precisions = multiclass_precision(gt, preds, tags)
            c_precisions.update((key, value * result_t) for key, value in c_precisions.items())
            results["classwise_precision"] = c_precisions

        if metric_ == "classwise_recall":
            c_recalls = multiclass_recall(gt, preds, tags)
            c_recalls.update((key, value * result_t) for key, value in c_recalls.items())
            results["classwise_recall"] = c_recalls

        if metric_ == "report" in eval_metrics:
            print(classification_report(gt, preds, target_names = tags))

        if type(metric_) != str and metric_.__bases__[0] == xMetrics:
            user_metric = metric_(gt, preds, tags)
            results[str(metric_.__name__)] = user_metric()
    return results

def multiclass_f1(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]],
        tags: List[str]
) -> dict:
    MODE = "classwise_f1"
    output_dict = classification_report(y_true, y_pred, target_names = tags, output_dict = True)
    f1s = {}
    for tag_name in output_dict.keys():
        if tag_name in tags:
            f1s[tag_name] = output_dict[tag_name]["f1-score"]
    return f1s

def multiclass_recall(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]],
        tags: List[str]
) -> dict:
    MODE = "classwise_recall"
    output_dict = classification_report(y_true, y_pred, target_names = tags, output_dict = True)
    recalls = {}
    for tag_name in output_dict.keys():
        if tag_name in tags:
            recalls[tag_name] = output_dict[tag_name]["recall"]
    return recalls

def multiclass_precision(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]],
        tags: List[str]
) -> dict:
    MODE = "classwise_precision"
    output_dict = classification_report(y_true, y_pred, target_names = tags, output_dict = True)
    precisions = {}
    for tag_name in output_dict.keys():
        if tag_name in tags:
            precisions[tag_name] = output_dict[tag_name]["precision"]
    return precisions

def recall(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]]
) -> dict:
    MODE = "avg_recall"
    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    recall_w = recall_score(y_true, y_pred, average="weighted")
    return {"weigted": recall_w,
            "micro": recall_micro,
            "macro": recall_macro
    }

def precision(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]]
) -> dict:
    MODE = "avg_precision"
    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    precision_w = precision_score(y_true, y_pred, average="weighted")
    return {"weighted": precision_w,
            "micro": precision_micro,
            "macro": precision_macro
    }

def f1(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]]
) -> dict:
    MODE = "avg_f1"
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    return {"weighted": f1_w,
            "micro": f1_micro,
            "macro": f1_macro
    }

def accuracy(
        y_true: Union[np.ndarray, List[List[int]]],
        y_pred: Union[np.ndarray, List[List[int]]]
) -> int:
    MODE = "acc"
    acc = accuracy_score(y_true, y_pred)
    return acc

def tag2onehot(
        preds: List[str],
        ground_truth: List[str],
        tags: List[str]
):
    digit_pred = np.array([tags.index(tag) for tag in preds])
    digit_gt = np.array([tags.index(tag) for tag in ground_truth])

    preds_onehot = np.zeros((digit_pred.size, len(tags)))
    gt_onehot = np.zeros((digit_gt.size, len(tags)))

    preds_onehot[np.arange(digit_pred.size), digit_pred] = 1
    gt_onehot[np.arange(digit_gt.size), digit_gt] = 1

    return preds_onehot, gt_onehot
    
