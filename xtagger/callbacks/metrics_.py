import warnings
from typing import Any, List, Optional, Dict

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class BaseMetric:
    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: Optional[List[str]] = None
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.tags = tags

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class ClasswiseF1(BaseMetric):
    mode: str = "classwise_f1"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        output_dict = classification_report(
            self.y_true, self.y_pred, target_names=self.tags, output_dict=True
        )
        f1s = {}
        for tag_name in output_dict.keys():
            if tag_name in self.tags:
                f1s[tag_name] = output_dict[tag_name]["f1-score"]
        return f1s


class ClasswiseRecall(BaseMetric):
    mode: str = "multiclass_f1"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        output_dict = classification_report(
            self.y_true, self.y_pred, target_names=self.tags, output_dict=True
        )
        recalls = {}
        for tag_name in output_dict.keys():
            if tag_name in self.tags:
                recalls[tag_name] = output_dict[tag_name]["recall"]
        return recalls


class ClasswisePrecision(BaseMetric):
    mode: str = "multiclass_precision"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        output_dict = classification_report(
            self.y_true, self.y_pred, target_names=self.tags, output_dict=True
        )
        precisions = {}
        for tag_name in output_dict.keys():
            if tag_name in self.tags:
                precisions[tag_name] = output_dict[tag_name]["precision"]
        return precisions


class Recall(BaseMetric):
    mode: str = "recall"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        recall_micro = recall_score(self.y_true, self.y_pred, average="micro")
        recall_macro = recall_score(self.y_true, self.y_pred, average="macro")
        recall_w = recall_score(self.y_true, self.y_pred, average="weighted")
        return {"weigted": recall_w, "micro": recall_micro, "macro": recall_macro}


class Precision(BaseMetric):
    mode: str = "precision"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        precision_micro = precision_score(self.y_true, self.y_pred, average="micro")
        precision_macro = precision_score(self.y_true, self.y_pred, average="macro")
        precision_w = precision_score(self.y_true, self.y_pred, average="weighted")
        return {"weighted": precision_w, "micro": precision_micro, "macro": precision_macro}


class F1(BaseMetric):
    mode: str = "f1"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        f1_micro = f1_score(self.y_true, self.y_pred, average="micro")
        f1_macro = f1_score(self.y_true, self.y_pred, average="macro")
        f1_w = f1_score(self.y_true, self.y_pred, average="weighted")
        return {"weighted": f1_w, "micro": f1_micro, "macro": f1_macro}


class Accuracy(BaseMetric):
    mode: str = "accuracy"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        acc = accuracy_score(self.y_true, self.y_pred)
        return {"accuracy": acc}
    

class ClassificationReport(BaseMetric):
    mode: str = "classification_report"

    def __init__(self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> None:
        print(classification_report(self.y_true, self.y_pred, target_names=self.tags))