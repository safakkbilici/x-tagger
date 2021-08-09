import xtagger
import torch
import numpy as np
from typing import List, Optional, Tuple

def check_monitor_metrics(metric):
    if metric == None:
        return
    else:
        if (metric not in xtagger.MONITOR) and ('_'.join(metric.split("_")[:-1]) not in xtagger.MONITOR):
            if type(metric) == str:
                raise ValueError(f'"{metric}" not found in monitor metrics, current monitor metrics are: {xtagger.MONITOR}')
            elif type(metric) != str and metric.__bases__[0] != xMetrics:
                raise ValueError(f'User defined metric {metric} must be inherited from xtagger.utils.metrics.xMetrics')

def check_monitor_eval_metrics(monitor, metrics):
    already = ["eval_loss", "train_loss"]
    if (monitor not in metrics) and \
       (monitor not in already) and \
       ('_'.join(monitor.split("_")[1:]) not in metrics) and \
       ('_'.join(monitor.split("_")[1:-1]) not in metrics):
            raise ValueError(f"Please add also {monitor} to eval_metrics argument.")

class Checkpointing():
    def __init__(
            self,
            model_path: str,
            model_name: str,
            monitor: Optional[str] = None,
            mode: str = "maximize",
            save_best: bool = True,
            verbose: int = 0
    ) -> None:
        r"""
        Args:
            model_path: directory for checkpoint.

            model_name: checkpoint file name for model.

            monitor: monitored metric name. If nıne and save_best is false,
                     than saves the model at every epoch.

            verbose: for printing monitored metric value at every epoch.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.verbose = verbose
        self.history = []

        if self.mode not in ["maximize", "minimize"]:
            raise ValueError("mode: maximize or minimize")

        if self.mode == "maximize":
            self.f = np.argmax
        else:
            self.f = np.argmin

        check_monitor_metrics(self.monitor)

    def save(self, model):
        torch.save(model.state_dict(), self.model_path+self.model_name)
    def load(self, model):
        model.model.load_state_dict(torch.load(self.model_path+self.model_name))
        model.model.eval()
        return model

    def save_in(self, model, results):
        if self.save_best:
            if self.monitor.split("_")[-1] == "loss":
                self.history.append(results[self.monitor])
            elif self.monitor.split("_")[-1] == "acc":
                self.history.append(results[self.monitor.split("_")[0]][self.monitor.split("_")[1]])
            else:
                if len(self.monitor.split("_")) == 3:
                    save_type = self.monitor.split("_")[0]
                    metric = '_'.join(self.monitor.split("_")[1:])
                    self.history.append(results[save_type][metric])
                elif len(self.monitor.split("_")) == 4:
                    save_type = self.monitor.split("_")[0]
                    metric = '_'.join(self.monitor.split("_")[1:-1])
                    self.history.append(results[save_type][metric][self.monitor.split("_")[-1]])

            if self.monitor.split("_")[-1] in ["loss", "acc"]:
                if self.monitor.split("_")[-1] == "loss":
                    monitor_metric = results[self.monitor]
                elif self.monitor.split("_")[-1] == "acc":
                    monitor_metric = results[self.monitor.split("_")[0]][self.monitor.split("_")[1]]
                
                if self.history[self.f(self.history)] == monitor_metric:
                    self.save(model)
                    if self.verbose:
                        print(f"Model is saved with {self.monitor} = {monitor_metric}")
            else:
                if len(self.monitor.split("_")) == 3:
                    save_type = self.monitor.split("_")[0]
                    metric = '_'.join(self.monitor.split("_")[1:])
                    monitor_metric = results[save_type][metric]
                elif len(self.monitor.split("_")) == 4:
                    save_type = self.monitor.split("_")[0]
                    metric = '_'.join(self.monitor.split("_")[1:-1])
                    monitor_metric = results[save_type][metric][self.monitor.split("_")[-1]]
                    
                if self.history[self.f(self.history)] == monitor_metric:
                    self.save(model)
                    if self.verbose:
                        print(f"Model is saved with {self.monitor} = {monitor_metric}")
        else:
            self.save(model)
