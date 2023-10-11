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
    results = {}

    for metric in eval_metrics:
        results[metric.mode] = metric(y_true, y_pred, tags)()
    return results


def write_results(results, output_dir):
    if not os.path.isdir(makepath(output_dir, "eval")):
        os.mkdir(makepath(output_dir, "eval"))

    with open(makepath(output_dir, "eval", "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=xtagger.GLOBAL_INDENT)
