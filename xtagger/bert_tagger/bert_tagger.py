import numpy as np
import time
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from xtagger.bert_tagger.bert import BERT
from xtagger.utils import metrics
from xtagger.utils import callbacks

import warnings
warnings.filterwarnings("ignore")

class BERTForTagging(object):
    def __init__(self, output_dim, TEXT, TAGS, dropout, device, model_name = "bert-base-cased", dont_stop_pretraining=False, cuda=True):
        self.output_dim = output_dim
        self.TEXT = TEXT
        self.TAGS = TAGS
        self.dropout = dropout
        self.model_name = model_name
        self.dont_stop_pretraining = dont_stop_pretraining
        self.device = device
        self.TAG_PAD_IDX = self.TAGS.vocab.stoi[self.TAGS.pad_token]

        if cuda and self.device.type=="cpu":
            print("can't see cuda, automatically using cpu.")

        self._build_model()

    def _build_model(self):
        self.model = BERT(
            self.model_name,
            self.output_dim,
            self.dropout,
            self.dont_stop_pretraining
        ).to(self.device)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad == True)
        nontrainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad == False)
        print(f'The model has {trainable} trainable parameters')
        print(f'The model has {nontrainable} non-trainable parameters')

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def fit(self, train_set, test_set, epochs=10, eval_metrics = ["acc"], result_type = "%", checkpointing = None):
        self.train_set = train_set
        self.test_set = test_set
        self._metrics = eval_metrics
        self.result_type = result_type
        
        metrics.check_eval_metrics(self._metrics)

        if checkpointing != None and checkpointing.save_best == True:
            callbacks.check_monitor_eval_metrics(checkpointing.monitor, eval_metrics)

        total = len(train_set) * epochs
        with tqdm(total = total) as tt:
            for epoch in range(epochs):
                for batch in self.train_set:
                    self.model.train()
                    text = batch.sentence.permute(1,0)
                    tags = batch.tags.permute(1,0)

                    self.optimizer.zero_grad()

                    out = self.model.forward(text)
                    loss = self.criterion(out.permute(0,2,1).float(), tags.long())
                    loss.backward()
                    self.optimizer.step()
                    tt.update()
                print("Evaluating...")
                results = self._eval()
                if checkpointing != None:
                    checkpointing.save_in(self.model, results)
                print(results)

    def evaluate(self, test_set=None, eval_metrics = ["acc"], result_type = "%"):
        if test_set == None:
            test_set = self.test_set

        metrics.check_eval_metrics(eval_metrics)
        test_y_pred = []
        test_y_true = []
        
        total = len(test_set)
        with tqdm(total = total) as ee:
            for batch in test_set:
                text = batch.sentence.permute(1,0)
                tags = batch.tags.permute(1,0)
                with torch.no_grad():
                    self.model.eval()
                    out = self.model.forward(text)
                    preds = torch.argmax(out, dim=-1).squeeze(dim=0).flatten()

                    tag_sample = tags.contiguous().view(-1)
                    non_pad_elements = (tag_sample != self.TAG_PAD_IDX).nonzero()
                    non_pad_preds = preds[non_pad_elements].squeeze(dim=-1)
                    non_pad_targets = tag_sample[non_pad_elements].squeeze(dim=-1)
                    test_y_pred.extend([self.TAGS.vocab.itos[a] for a in non_pad_preds])
                    test_y_true.extend([self.TAGS.vocab.itos[a] for a in non_pad_targets])
                    ee.update()
                     
                     
        test_preds_oh, test_gt_oh = metrics.tag2onehot(test_y_pred, test_y_true, self.TAGS.vocab.itos)
        results = metrics.metric_results(
            test_gt_oh[1:],
            test_preds_oh[1:],
            eval_metrics,
            result_type,
            self.TAGS.vocab.itos[1:]
        )

        return results

    def _eval(self):
        total = len(self.train_set) + len(self.test_set)
        eval_loss, eval_count = 0, 0
        train_loss, train_count = 0, 0

        test_y_pred = []
        test_y_true = []
        
        train_y_pred = []
        train_y_true = []
        
        with tqdm(total = total) as ee:
            for batch in self.test_set:
                text = batch.sentence.permute(1,0)
                tags = batch.tags.permute(1,0)
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

            for batch in self.train_set:
                text = batch.sentence.permute(1,0)
                tags = batch.tags.permute(1,0)
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
            self._metrics,
            self.result_type,
            self.TAGS.vocab.itos
        )

        results["train"] = metrics.metric_results(
            train_gt_oh,
            train_preds_oh,
            self._metrics,
            self.result_type,
            self.TAGS.vocab.itos
        )

        results["eval_loss"] = eval_loss
        results["train_loss"] = train_loss
        return results


    def predict(self, sentence, tokenizer):
        self.model.eval()
        if isinstance(sentence, str):
            tokens = tokenizer.tokenize(sentence)
        else:
            tokens = sentence

        if self.TEXT.lower:
            tokens = [t.lower() for t in tokens]


        numericalized_tokens = tokenizer.convert_tokens_to_ids(tokens)
        numericalized_tokens = [self.TEXT.init_token] + numericalized_tokens

        unk_token_id = self.TEXT.unk_token
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_token_id]
        token_tensor = torch.LongTensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(0).to(self.device)

        predictions = self.model(token_tensor)
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.TAG.vocab.itos[t.item()] for t in top_predictions]

        predicted_tags = predicted_tags[1:]
        assert len(tokens) == len(predicted_tags)
        return list(zip(tokens, predicted_tags)), unks
    
    def configure_optimizer(self, optimizer: torch.optim, **args):
        self.optimizer = optimizer(self.model.parameters(), **args)

    def configure_loss_fn(self, loss_fn: torch.nn, **args):
        self.criterion = loss_fn(**args)

    def get_model_details(self):
        return {"model": self.model, "optimizer": self.optimizer, "loss function": self.criterion}
     
