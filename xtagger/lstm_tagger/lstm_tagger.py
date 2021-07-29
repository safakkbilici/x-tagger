import spacy
import torch
import numpy as np
import time
import random
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from xtagger.utils.time_utils import epoch_time
from xtagger.lstm_tagger.lstm import LSTM
from xtagger.utils import metrics
from xtagger.utils import callbacks

class LSTMForTagging(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 TEXT,
                 TAGS,
                 embedding_dim=100,
                 hidden_dim=128,
                 n_layers = 2,
                 bidirectional=True,
                 dropout=0.25,
                 cuda=True,
                 tag_pad_idx = None,
                 pad_idx = None,
                 
    ):


        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.TEXT = TEXT
        self.TAGS = TAGS
        
        if tag_pad_idx == None:
            self.TAG_PAD_IDX = self.TAGS.vocab.stoi[self.TAGS.pad_token]
        else:
            self.TAG_PAD_IDX = tag_pad_idx
        if pad_idx == None:
            self.TEXT_PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]
        else:
            self.TEXT_PAD_IDX = text_pad_idx

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cuda and self.device.type=="cpu":
            print("can't see cuda, automatically using cpu.")

        self.build_model()


    def init_weigths(self, m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def build_model(self):
        self.model = LSTM(
            self.input_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.n_layers,
            self.bidirectional,
            self.dropout,
            self.TEXT_PAD_IDX
        )

        self.model.apply(self.init_weigths)
        self.model = self.model.to(self.device)
        print(f'The model has {self.count_parameters():,} trainable parameters')
        #self.TAG_PAD_IDX = self.TAGS.vocab.stoi[self.TAGS.pad_token]

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.TAG_PAD_IDX)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())


    def fit(self, train_set, test_set, epochs=10, save_name = "lstm_model.pt", eval_metrics = ["acc"], result_type = "%", checkpointing = None):
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
            test_gt_oh,
            test_preds_oh,
            eval_metrics,
            result_type,
            self.TAGS.vocab.itos
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
                    #for text_sample, tag_sample in zip(text, tags):
                    self.model.eval()
                    eval_count += 1
                    #out = self.model.forward(text[None,:])
                    out = self.model.forward(text)
                    preds = torch.argmax(out, dim=-1).squeeze(dim=0).flatten()
                    loss = self.criterion(out.permute(0,2,1).float(), tags.long()) #tags[None, :]
                    
                    eval_loss += loss.item()
                    #non_pad_elements = (tag_sample != self.TAG_PAD_IDX).nonzero()
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
                    #for text_sample, tag_sample in zip(text, tags):
                    self.model.eval()
                    train_count += 1
                    #out = self.model.forward(text[None,:])
                    out = self.model.forward(text)
                    preds = torch.argmax(out, dim=-1).squeeze(dim=0).flatten()
                    loss = self.criterion(out.permute(0,2,1).float(), tags.long()) #tags[None, :]

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

    def predict(self, sentence):
        self.model.eval()
        if isinstance(sentence, str):
            nlp = spacy.load("en_core_web_sm")
            tokens = [token.text for token in nlp(sentence)]
        else:
            tokens = [token for token in sentence]

        if self.TEXT.lower:
            tokens = [t.lower() for t in tokens]

        numericalized_tokens = [self.TEXT.vocab.stoi[t] for t in tokens]
        unk_idx = self.TEXT.vocab.stoi[self.TEXT.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        token_tensor = torch.LongTensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(0).to(self.device)
        predictions = self.model(token_tensor).squeeze(0)
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.TAGS.vocab.itos[t.item()] for t in top_predictions]
        return list(zip(tokens, predicted_tags)), unks
    


    
