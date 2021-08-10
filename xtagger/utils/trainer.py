import torch
from xtagger.utils import metrics
import torch.nn as nn
from tqdm.auto import tqdm

class PyTorchTagTrainer():
    def __init__(self,model, optimizer, criterion, train_iterator, val_iterator, test_iterator, TEXT, TAGS, device, eval_metrics = ["acc"], checkpointing = None, result_type = "%"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator
        
        if self.test_iterator == None:
            self.test_iterator = self.val_iterator
        
        self.TEXT = TEXT
        self.TAGS = TAGS
        self.TAG_PAD_IDX = self.TAGS.vocab.stoi[self.TAGS.pad_token]
        self.device = device
        self.result_type = result_type
        
        metrics.check_eval_metrics(eval_metrics)
        self.eval_metrics = eval_metrics
        
        self.checkpointing = checkpointing
        if self.checkpointing != None and checkpointing.save_best == True:
            callbacks.check_monitor_eval_metrics(checkpointing.monitor, self.eval_metrics)

    def train(self, epochs = 10):
        total = len(self.train_iterator) * epochs
        batch_size = self.train_iterator.batch_size
        if next(iter(self.train_iterator)).sentence.size(0) != batch_size:
            self.batch_first = False
            
        with tqdm(total = total) as tt:
            for epoch in range(epochs):
                for batch in self.train_iterator:
                    self.model.train()
                    if not self.batch_first:
                        text = batch.sentence.permute(1,0).to(self.device)
                        tags = batch.tags.permute(1,0).to(self.device)

                    self.optimizer.zero_grad()
                    
                    out = self.model.forward(text)
                    loss = self.criterion(out.permute(0,2,1).float(), tags.long())
                    loss.backward()
                    self.optimizer.step()
                    tt.update()
                print("Evaluating...")
                results = self._eval()
                print(results)

        return self.model

    def _eval(self):
        total = len(self.train_iterator) + len(self.val_iterator)
        eval_loss, eval_count = 0, 0
        train_loss, train_count = 0, 0
        test_y_pred = []
        test_y_true = []
        train_y_pred = []
        train_y_true = []

        with tqdm(total = total) as ee:
            for batch in self.val_iterator:
                if not self.batch_first:
                    text = batch.sentence.permute(1,0).to(self.device)
                    tags = batch.tags.permute(1,0).to(self.device)
                    
                with torch.no_grad():
                    self.model.eval()
                    eval_count += 1
                    out = self.model.forward(text)
                    preds = torch.argmax(out, dim=-1).squeeze(dim=0).flatten()
                    loss = self.criterion(out.permute(0,2,1).float(), tags.long())
                    eval_loss += loss.item()
                    tag_sample = tags.contiguous().view(-1)
                    non_pad_elements = (tag_sample != self.TAG_PAD_IDX).nonzero()
                    non_pad_preds = preds[non_pad_elements].squeeze(dim=-1)
                    non_pad_targets = tag_sample[non_pad_elements].squeeze(dim=-1)
                    test_y_pred.extend([self.TAGS.vocab.itos[a] for a in non_pad_preds])
                    test_y_true.extend([self.TAGS.vocab.itos[a] for a in non_pad_targets])
                    ee.update()

            for batch in self.train_iterator:
                if not self.batch_first:
                    text = batch.sentence.permute(1,0).to(self.device)
                    tags = batch.tags.permute(1,0).to(self.device)
                with torch.no_grad():
                    self.model.eval()
                    train_count += 1
                    out = self.model.forward(text)
                    preds = torch.argmax(out, dim=-1).squeeze(dim=0).flatten()
                    loss = self.criterion(out.permute(0,2,1).float(), tags.long())
                    train_loss += loss.item()
                    tag_sample = tags.contiguous().view(-1)
                    non_pad_elements = (tag_sample != self.TAG_PAD_IDX).nonzero()
                    non_pad_preds = preds[non_pad_elements].squeeze(dim=-1)
                    non_pad_targets = tag_sample[non_pad_elements].squeeze(dim=-1)
                    train_y_pred.extend([self.TAGS.vocab.itos[a] for a in non_pad_preds])
                    train_y_true.extend([self.TAGS.vocab.itos[a] for a in non_pad_targets])
                    ee.update()

        train_loss = train_loss / train_count
        eval_loss = eval_loss / eval_count

        test_preds_oh, test_gt_oh = metrics.tag2onehot(test_y_pred, test_y_true, self.TAGS.vocab.itos)
        train_preds_oh, train_gt_oh = metrics.tag2onehot(train_y_pred, train_y_true, self.TAGS.vocab.itos)

        results = {}
        results["eval"] = metrics.metric_results(
            test_gt_oh,
            test_preds_oh,
            self.eval_metrics,
            self.result_type,
            self.TAGS.vocab.itos
        )

        results["train"] = metrics.metric_results(
            train_gt_oh,
            train_preds_oh,
            self.eval_metrics,
            self.result_type,
            self.TAGS.vocab.itos
        )

        results["eval_loss"] = eval_loss
        results["train_loss"] = train_loss
        return results


    def evaluate(self):
        test_y_pred = []
        test_y_true = []
        total = len(self.test_iterator)

        test_loss, test_count = 0, 0
        with tqdm(total = total) as ee:
            for batch in self.test_iterator:
                if not self.batch_first:
                    text = batch.sentence.permute(1,0).to(self.device)
                    tags = batch.tags.permute(1,0).to(self.device)
                    with torch.no_grad():
                        self.model.eval()
                        test_count += 1
                        out = self.model.forward(text)

                        preds = torch.argmax(out, dim=-1).squeeze(dim=0).flatten()
                        loss = self.criterion(out.permute(0,2,1).float(), tags.long())
                        
                        test_loss += loss.item()
                        
                        tag_sample = tags.contiguous().view(-1)
                        non_pad_elements = (tag_sample != self.TAG_PAD_IDX).nonzero()
                        non_pad_preds = preds[non_pad_elements].squeeze(dim=-1)
                        non_pad_targets = tag_sample[non_pad_elements].squeeze(dim=-1)
                        test_y_pred.extend([self.TAGS.vocab.itos[a] for a in non_pad_preds])
                        test_y_true.extend([self.TAGS.vocab.itos[a] for a in non_pad_targets])
                        ee.update()

        test_loss = test_loss / test_count
        test_preds_oh, test_gt_oh = metrics.tag2onehot(test_y_pred, test_y_true, self.TAGS.vocab.itos)

        results = {}
        results["eval"] = metrics.metric_results(
            test_gt_oh,
            test_preds_oh,
            self.eval_metrics,
            self.result_type,
            self.TAGS.vocab.itos
        )

        results["eval_loss"] = test_loss
        return results
