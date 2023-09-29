from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import xtagger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from xtagger.callbacks.checkpoint import Checkpointing
from xtagger.callbacks.metrics import convert_to_onehot, metric_results
from xtagger.callbacks.metrics_ import Accuracy, BaseMetric
from xtagger.tokenization.base import TokenizerBase
from xtagger.utils.data import LabelEncoder, convert_to_dataloader


class RNNTagger(nn.Module):
    def __init__(
        self,
        rnn: str,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: int,
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        rnn = rnn.upper()

        self.rnn = rnn
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx
        )

        assert self.rnn in ["LSTM", "GRU", "RNN"], "Only lstm, rnn, and gru is available."
        RNN = getattr(nn, rnn)

        self.model = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.embedding(input_ids))
        if self.rnn == "lstm":
            out, (hidden, cell) = self.model(out)
        else:
            out, hidden = self.model(out)

        out = self.fc(self.dropout(out))
        return out

    def fit(
        self,
        train_set: Union[List[List[Tuple[str, str]]], DataLoader],
        dev_set: Union[List[List[Tuple[str, str]]], DataLoader],
        tokenizer: TokenizerBase,
        label_encoder: LabelEncoder,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device: torch.device,
        eval_metrics: List[BaseMetric] = [Accuracy],
        callback: Optional[Checkpointing] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        pretokenizer: Callable = lambda x: x,
    ):
        if type(train_set) != DataLoader:
            train_dataloader = convert_to_dataloader(
                dataset=train_set,
                tokenizer=tokenizer,
                label_encoder=label_encoder,
                batch_size=batch_size,
                max_length=max_length,
                pretokenizer=pretokenizer,
                shuffle=True,
            )

        if type(dev_set) != DataLoader:
            dev_dataloader = convert_to_dataloader(
                dataset=train_set,
                tokenizer=tokenizer,
                label_encoder=label_encoder,
                batch_size=batch_size,
                max_length=max_length,
                pretokenizer=pretokenizer,
                shuffle=False,
            )

        total = len(train_dataloader) * num_epochs
        results = defaultdict(lambda: defaultdict(dict))

        with tqdm(
            total=total, desc="Training", disable=xtagger.DISABLE_PROGRESS_BAR
        ) as progressbar:
            for epoch in range(num_epochs):
                total_loss = 0
                batch_count = 0
                for idx, batch in enumerate(train_dataloader):
                    self.train()

                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    out = self.forward(input_ids=input_ids)
                    loss = criterion(out.permute(0, 2, 1).float(), labels.long())
                    ## TODO: add mixed precision training here
                    ## TODO: add gradient scaling here
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1

                    progressbar.update()

                eval_results, eval_loss = self.__eval(
                    dev_dataloader=dev_dataloader,
                    criterion=criterion,
                    device=device,
                    label_encoder=label_encoder,
                    eval_metrics=eval_metrics,
                )

                results["train"]["loss"] = total_loss / batch_count
                results["eval"]["loss"] = eval_loss
                results["eval"] = eval_results

                callback(
                    model=self, results=results, path=".", name="model", indicator_name=str(epoch)
                )

                if scheduler != None:
                    scheduler.step()

        return results

    @torch.no_grad()
    def __eval(
        self,
        dev_dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        label_encoder: LabelEncoder,
        eval_metrics: List[BaseMetric] = [Accuracy],
    ):
        y_true_test = []
        y_pred_test = []
        total_loss = 0
        batch_count = 0

        total = len(dev_dataloader)
        with tqdm(
            total=total, desc="Evaluating", disable=xtagger.DISABLE_PROGRESS_BAR
        ) as progressbar:
            for idx, batch in enumerate(dev_dataloader):
                self.eval()

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                out = self.forward(input_ids=input_ids)
                loss = criterion(out.permute(0, 2, 1).float(), labels.long())

                preds = torch.softmax(dim=-1).argmax(out, dim=-1).squeeze(dim=0).flatten()

                ground_truth = labels.contiguous().view(-1)
                non_pad_indices = (ground_truth != label_encoder.pad_tag_id).nonzero()
                non_pad_preds = preds[non_pad_indices].squeeze(dim=-1)
                non_pad_ground_truth = ground_truth[non_pad_indices].squeeze(dim=-1)

                y_pred_test.extend([label_encoder[t] for t in non_pad_preds])
                y_true_test.extend([label_encoder[t] for t in non_pad_ground_truth])

                total_loss += loss.item()
                batch_count += 1

                progressbar.update()

        y_pred_test, y_true_test = convert_to_onehot(
            y_pred_test, y_true_test, tags=label_encoder.reverse_maps
        )

        results = metric_results(
            y_pred=y_pred_test,
            y_true=y_true_test,
            eval_metrics=eval_metrics,
            tags=label_encoder.reverse_maps,
        )
        total_loss = total_loss / batch_count

        return results, total_loss

    def evaluate(
        self,
        test_set: Union[List[List[Tuple[str, str]]], DataLoader],
        tokenizer: TokenizerBase,
        criterion: Optional[nn.Module] = None,
    ):
        pass

    def predict(self, sentence: Union[str, List[str]], tokenizer: TokenizerBase):
        pass