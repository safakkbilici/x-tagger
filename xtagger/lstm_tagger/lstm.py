import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_layers: int,
            bidirectional: bool,
            dropout: float,
            pad_idx: int
    ) -> None:

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            dropout = dropout if n_layers > 1 else 0,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        preds= self.fc(self.dropout(outputs))
        return preds
