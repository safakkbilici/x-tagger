import logging
from typing import Literal

import torch
from xtagger.utils.helpers import makepath

logger = logging.getLogger(__name__)


class Checkpointing:
    def __init__(
        self, on: str = "accuracy", mode: Literal["min", "max"] = "max",
        strategy: Literal["best", "every", "none"] = "best", verbose: int = 0
    ) -> None:
        """Helper class for tracking your metrics and checkpoints

        Args:
            on (str): tracked metric for saving models
            mode (Literal["min", "max"]): optimal metric at mode
            strategy (Literal["best", "every", "none"]): save best model, save every model, or none
            verbose (int): enable logging
        """
        self.on = on
        self.mode = mode
        self.strategy = strategy
        self.verbose = verbose
        self.history = []

        if self.mode not in ["max", "min"]:
            raise ValueError("mode: max or min")

        if self.strategy not in ["best", "every", "none"]:
            raise ValueError("mode: best, every or none")

        if self.mode == "max":
            self.f = max
        else:
            self.f = min

    @staticmethod
    def save_torch(model: torch.nn.Module, to: str) -> None:
        """Saves the torch model

        Args:
            model (torch.nn.Module): torch model
            to (str): path to save
        """
        torch.save(model.state_dict(), to)

    @staticmethod
    def load(model: torch.nn.Module, to: str) -> torch.nn.Module:
        """Loads the torch model

        Args:
            model (torch.nn.Module): torch model
            to (str): the path and file name joined you want to load

        Returns:
            model (torch.nn.Module): the loaded model
        """
        model.load_state_dict(torch.load(to))
        return model

    def save(
        self, model: torch.nn.Module, results: dict, path: str, name: str, indicator_name: str = ""
    ) -> None:
        """Saves the torch model

        Args:
            model (torch.nn.Module): torch model
            results (dict): your on-the-fly metric value dictionary
            path (str): path to save
            name (str): model name
            indicator_name (str): for saving strategies, you can give a suffix indicator to distinguish
        """
        base_metric = self.on
        if "." in self.on:
            base_metric = self.on.split(".")[0]
            avg_metric = self.on.split(".")[1]
            result = results["eval"][base_metric][avg_metric]

        else:
            result = results["eval"][base_metric]

        self.history.append(result)

        best_model = False
        if self.f(self.history) == result:
            best_model = True

        if self.strategy == "best" and best_model:
            to = makepath(path, name + "_best.pt")
            self.save_torch(model, to)

            if self.verbose:
                logger.info(f"Model is saved to, with {self.on}={result}, to {to}.")

        if self.strategy == "every":
            to = makepath(path, name + "_" + str(indicator_name) + ".pt")
            self.save_torch(model, to)

            if self.verbose:
                logger.info(f"Model is saved to, with {self.on}={result}, to {to}.")
