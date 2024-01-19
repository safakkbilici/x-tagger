import json
import os
import warnings
from typing import Dict, List, Union

import numpy as np
import xtagger
from xtagger.callbacks.metrics_ import *
from xtagger.utils.helpers import makepath

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def metric_results(
    y_true: Union[np.ndarray, List[List[int]]],
    y_pred: Union[np.ndarray, List[List[int]]],
    eval_metrics: List[BaseMetric],
    tags: List[str],
) -> Dict[str, Dict[str, float] | float]:
    """Computes selected metrics regarding given y_true and y_pred

    Args:
        y_true (List[float]): list of your target values
        y_pred (List[float]): list of your predicted values
        eval_metrics (List[BaseMetric]): list of eval metric class
        tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred

    Returns:
        results (Dict[str, Dict[str, float] | float]): dictionary of metric values with their names
    """
    results = {}

    for metric in eval_metrics:
        results[metric.mode] = metric(y_true, y_pred, tags)()
    return results


def write_results(results: Dict[str, Dict[str, float] | float], output_dir: str):
    """Writes metric results into output_dir/eval/results.json

    Args:
        results (Dict[str, Dict[str, float] | float]): computed metrics
        output_dir (str): directory to save
    """
    if not os.path.isdir(makepath(output_dir, "eval")):
        os.mkdir(makepath(output_dir, "eval"))

    with open(makepath(output_dir, "eval", "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=xtagger.GLOBAL_INDENT)
