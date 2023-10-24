import logging
from typing import List

import torch
from xtagger.utils.helpers import makepath

logger = logging.getLogger(__name__)


class Checkpointing:
    def __init__(
        self, on: str = "accuracy", mode: str = "max", strategy: str = "best", verbose: int = 0
    ) -> None:
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

    def save(self, model: torch.nn.Module, to: str):
        torch.save(model.state_dict(), to)

    @staticmethod
    def load(model: torch.nn.Module, to: str):
        model.load_state_dict(torch.load(to))
        return model

    def save(
        self, model: torch.nn.Module, results: dict, path: str, name: str, indicator_name: str = ""
    ) -> None:
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
            self.save(model, to)

            if self.verbose:
                logger.info(f"Model is saved to, with {self.on}={result}, to {to}.")

        if self.strategy == "every":
            to = makepath(path, name + "_" + str(indicator_name) + ".pt")
            self.save(model, to)

            if self.verbose:
                logger.info(f"Model is saved to, with {self.on}={result}, to {to}.")
