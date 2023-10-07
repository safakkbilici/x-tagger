from typing import Union

import xtagger
from xtagger.callbacks.metrics_ import BaseMetric


def validate_eval_metrics(metrics: Union[str, BaseMetric]) -> None:
    if not metrics:
        raise ValueError("Metrics cannot be empty list.")
    else:
        for metric in metrics:
            if metric not in xtagger.IMPLEMENTED_METRICS:
                if type(metric) == str:
                    raise NotImplementedError(
                        f'"{metric}" not found in implemented metrics, current implemented metrics are: {xtagger.IMPLEMENTED_METRICS}'
                    )
            elif type(metric) != str and metric.__bases__[0] != BaseMetric:
                raise ValueError(
                    f"User defined metric {metric} must be inherited from xtagger.utils.metrics.xMetrics"
                )


def validate_monitor_metrics(metric):
    if metric == None:
        return
    else:
        if (metric not in xtagger.MONITOR) and (
            "_".join(metric.split("_")[:-1]) not in xtagger.MONITOR
        ):
            if type(metric) == str:
                raise ValueError(
                    f'"{metric}" not found in monitor metrics, current monitor metrics are: {xtagger.MONITOR}'
                )
            elif type(metric) != str and metric.__bases__[0] != BaseMetric:
                raise ValueError(
                    f"User defined metric {metric} must be inherited from xtagger.utils.metrics.xMetrics"
                )


def validate_monitor_eval_metrics(monitor, metrics):
    already = ["eval_loss", "train_loss"]
    if (
        (monitor not in metrics)
        and (monitor not in already)
        and ("_".join(monitor.split("_")[1:]) not in metrics)
        and ("_".join(monitor.split("_")[1:-1]) not in metrics)
    ):
        raise ValueError(f"Please add also {monitor} to eval_metrics argument.")


def validate_prior_tags(prior, model_tags):
    if prior == None:
        return
    elif type(prior) not in xtagger.IMPLEMENTED_REGEX_LANGUAGES:
        raise TypeError(
            f"The tagger must be [Language]RegExTagger, current implemented languages: {xtagger.IMPLEMENTED_REGEX_LANGUAGES}"
        )
    else:
        prior_tags = [pair[1] for pair in prior.get_patterns()]
        for tag in prior_tags:
            if tag not in model_tags:
                raise ValueError("Passing different tags from training set is ambigious.")


def validate_morphological_tags(morphological, model_tags):
    if morphological == None:
        return
    elif type(morphological) not in xtagger.IMPLEMENTED_REGEX_LANGUAGES:
        raise TypeError(
            f"The tagger must be [Language]RegExTagger, current implemented languages: {xtagger.IMPLEMENTED_REGEX_LANGUAGES}"
        )
    else:
        morphological_tags = [pair[1] for pair in morphological.get_patterns()]
        for tag in morphological_tags:
            if tag not in model_tags:
                raise ValueError("Passing different tags from training set is ambigious.")
