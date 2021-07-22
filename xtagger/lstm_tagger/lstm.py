import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers,
                            bidirectional = bidirectional, dropout = dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        print("input shape: ",text.shape)
        embedded = self.dropout(self.embedding(text))
        print("embedded shape: ",embedded.shape)
        outputs, (hidden, cell) = self.lstm(embedded)
        print("outputs shape: ",outputs.shape)
        preds= self.fc(self.dropout(outputs))
        print("preds shape: ",preds.shape)
        return preds
