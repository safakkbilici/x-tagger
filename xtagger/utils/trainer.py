import torch
from xtagger.utils import metrics
import torch.nn as nn

class Trainer():
    def __init__(model, train_iterator, val_iterator, TEXT, TAGS, device, eval_metrics = ["acc"], checkpointing = None):
        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.TEXT = TEXT
        self.TAGS = TAGS
        self.device = device
        
        metrics.check_eval_metrics(eval_metrics)
        self.eval_metrics = eval_metrics
        
        self.checkpointing = checkpointing
        if self.checkpointing != None and checkpointing.save_best == True:
            callbacks.check_monitor_eval_metrics(checkpointing.monitor, self.eval_metrics)

    def train(self, epochs = 10, result_type = "%"):
        raise NotImplementedError()

        
        
