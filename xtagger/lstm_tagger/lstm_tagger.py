import spacy
import torch
import numpy as np
import time
import random
import torch.nn as nn
import torch.optim as optim

from xtagger.utils.time_utils import epoch_time
from xtagger.lstm_tagger.lstm import LSTM

class LSTMForTagging(object):
    def __init__(self, TEXT, TAGS, embedding_dim=100, hidden_dim=128,
                 n_layers = 2, bidirectional=True, dropout=0.25, cuda=True):

        self.TEXT = TEXT
        self.TAGS = TAGS
        self.input_dim = len(self.TEXT.vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = len(self.TAGS.vocab)
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.pad_idx = self.TEXT.vocab.stoi[self.TEXT.pad_token]

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
        self.model = LSTM(self.input_dim, self.embedding_dim,
                          self.hidden_dim, self.output_dim,
                          self.n_layers, self.bidirectional,
                          self.dropout, self.pad_idx)

        self.model.apply(self.init_weigths)
        self.model = self.model.to(self.device)
        print(f'The model has {self.count_parameters():,} trainable parameters')
        self.TAG_PAD_IDX = self.TAGS.vocab.stoi[self.TAGS.pad_token]

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.TAG_PAD_IDX)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())


    def categorical_accuracy(self,preds, y, tag_pad_idx):
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        non_pad_elements = (y != tag_pad_idx).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


    def train_step(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()

        for batch in self.train_set:
            text = batch.sentence
            tags = batch.tags

            self.optimizer.zero_grad()
            predictions = self.model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = self.criterion(predictions, tags)
            acc = self.categorical_accuracy(predictions.cpu(), tags.cpu(), self.TAG_PAD_IDX)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_set), epoch_acc / len(self.train_set)

    def eval_step(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():

            for batch in self.test_set:

                text = batch.sentence
                tags = batch.tags

                predictions = self.model(text)

                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)

                loss = self.criterion(predictions, tags)

                acc = self.categorical_accuracy(predictions.cpu(), tags.cpu(), self.TAG_PAD_IDX)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.test_set), epoch_acc / len(self.test_set)

    def fit(self, train_set, test_set, epochs=10, save_name = "lstm_model.pt"):
        self.train_set = train_set
        self.test_set = test_set
        
        best_valid_loss = float('inf')
        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_step()
            valid_loss, valid_acc = self.eval_step()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), save_name)

            print(f"""Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc*100:.2f}%;
            Val. Loss: {valid_loss:.3f}, Val. Accuracy: {valid_acc*100:.2f}% | Time Taken: {epoch_mins}m {epoch_secs}s""")


    def load_best_model(self, save_name = "lstm_model.pt"):
        self.model.load_state_dict(torch.load(save_name))
        test_loss, test_acc = eval_step()
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')


    def predict(self,sentence):
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
        token_tensor = token_tensor.unsqueeze(-1).to(self.device)
        predictions = self.model(token_tensor)
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.TAGS.vocab.itos[t.item()] for t in top_predictions]

        return tokens, predicted_tags, unks
