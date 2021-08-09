import torch.nn as nn
from transformers import AutoModel

class BERT(nn.Module):
    def __init__(
            self,
            bert_name: str,
            output_dim: int,
            dropout: float,
            dont_stop_pretraining: bool = False
    ) -> None:
        super(BERT, self).__init__()
        self.bert_name = bert_name

        self.bert = AutoModel.from_pretrained(self.bert_name)
        embed_dim = self.bert.config.to_dict()["hidden_size"]

        if not dont_stop_pretraining:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        contextualized_repr = self.dropout(self.bert(x)["last_hidden_state"])
        out = self.fc_out(contextualized_repr)
        return out
        
