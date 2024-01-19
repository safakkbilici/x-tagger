import warnings
from typing import Any, Dict, List, Optional

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
        """Abstract class for implementing your metrics

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.tags = tags

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ClasswiseF1(BaseMetric):
    mode: str = "classwise_f1"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        """Computes class-wise f1 score

        Returns:
            result (Dict[str, float]): dict with value of class-wise f1 score, key of metric name
        """
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
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        """Computes class-wise recall score

        Returns:
            result (Dict[str, float]): dict with value of class-wise recall score, key of metric name
        """
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
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        """Computes class-wise precision score

        Returns:
            result (Dict[str, float]): dict with value of class-wise precision score, key of metric name
        """
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
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        """Computes recall score

        Returns:
            result (Dict[str, float]): dict with value of recall score, key of metric name
        """
        recall_micro = recall_score(self.y_true, self.y_pred, average="micro")
        recall_macro = recall_score(self.y_true, self.y_pred, average="macro")
        recall_w = recall_score(self.y_true, self.y_pred, average="weighted")
        return {"weigted": recall_w, "micro": recall_micro, "macro": recall_macro}


class Precision(BaseMetric):
    mode: str = "precision"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        """Computes precision score

        Returns:
            result (Dict[str, float]): dict with value of precision score, key of metric name
        """
        precision_micro = precision_score(self.y_true, self.y_pred, average="micro")
        precision_macro = precision_score(self.y_true, self.y_pred, average="macro")
        precision_w = precision_score(self.y_true, self.y_pred, average="weighted")
        return {"weighted": precision_w, "micro": precision_micro, "macro": precision_macro}


class F1(BaseMetric):
    mode: str = "f1"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> Dict[str, float]:
        """Computes f1 score

        Returns:
            result (Dict[str, float]): dict with value of f1 score, key of metric name
        """
        f1_micro = f1_score(self.y_true, self.y_pred, average="micro")
        f1_macro = f1_score(self.y_true, self.y_pred, average="macro")
        f1_w = f1_score(self.y_true, self.y_pred, average="weighted")
        return {"weighted": f1_w, "micro": f1_micro, "macro": f1_macro}


class Accuracy(BaseMetric):
    mode: str = "accuracy"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        """Implements abstract base metric class

        Args:
            y_true (List[float]): list of your target values
            y_pred (List[float]): list of your predicted values
            tags (Optional[List[str]]): list of tag names regarding values in y_true and y_pred
        """
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> float:
        """Computes accuracy score

        Returns:
            result (float): accuracy score value
        """
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc


class ClassificationReport(BaseMetric):
    mode: str = "classification_report"

    def __init__(
        self, y_true: List[float], y_pred: List[float], tags: List[str] | None = None
    ) -> None:
        super().__init__(y_true, y_pred, tags)

    def __call__(self) -> None:
        print(classification_report(self.y_true, self.y_pred, target_names=self.tags))
