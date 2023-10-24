import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import xtagger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from xtagger.callbacks.checkpoint import Checkpointing
from xtagger.callbacks.metrics import metric_results, write_results
from xtagger.callbacks.metrics_ import Accuracy, BaseMetric
from xtagger.tokenization.base import TokenizerBase
from xtagger.utils.data import LabelEncoder, convert_to_dataloader
from xtagger.utils.helpers import padded_argmax_and_flatten, to_string, to_tensor
from xtagger.utils.pooling import IdentityPooler, compute_dimension

logger = logging.getLogger(__name__)


class PretrainedEncoderTagger(nn.Module):
    def __init__(
        self,
        name: str,
        dropout: float,
        n_classes: int,
        freeze: bool = True,
        pooler: nn.Module = IdentityPooler(),
    ) -> None:
        from transformers import AutoModel

        super().__init__()

        self.pretrained_encoder = AutoModel.from_pretrained(name)
        self.config = self.pretrained_encoder.config
        if freeze:
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.pooler = pooler
        hidden_size = compute_dimension(
            kernel_size=self.pooler.kernel_size,
            stride=self.pooler.stride,
            hidden_size=self.config.hidden_size,
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        out = self.pretrained_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        out = out.last_hidden_state
        out = self.pooler(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def fit(
        self,
        train_set: Union[xtagger.DATASET_TYPE, DataLoader],
        dev_set: Union[xtagger.DATASET_TYPE, DataLoader],
        tokenizer: TokenizerBase,
        label_encoder: LabelEncoder,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device: torch.device,
        eval_metrics: List[BaseMetric] = [Accuracy],
        output_dir: str = "./out",
        use_amp: bool = False,
        callback: Optional[Checkpointing] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        pretokenizer: Callable = xtagger.DEFAULT_PRETOKENIZER,
    ):
        self.to(device)
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

        else:
            train_dataloader = train_set

        if type(dev_set) != DataLoader:
            dev_dataloader = convert_to_dataloader(
                dataset=dev_set,
                tokenizer=tokenizer,
                label_encoder=label_encoder,
                batch_size=batch_size,
                max_length=max_length,
                pretokenizer=pretokenizer,
                shuffle=False,
            )

        else:
            dev_dataloader = dev_set

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        logger.info(f"Output path is set to {output_dir}.")
        logger.info(f"Evaluation results will be saved to {output_dir}.")
        logger.info(f"Checkpoints will be saved to {output_dir}.")

        if use_amp:
            scaler = GradScaler()

        total = len(train_dataloader) * num_epochs
        results = defaultdict(lambda: defaultdict(dict))

        with tqdm(
            total=total,
            desc="Training",
            disable=xtagger.DISABLE_PROGRESS_BAR,
            position=0,
            leave=True,
        ) as progressbar:
            for epoch in range(num_epochs):
                total_loss = 0
                batch_count = 0
                for idx, batch in enumerate(train_dataloader):
                    self.train()

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()

                    if use_amp:
                        with self._autocast():
                            out = self.forward(
                                input_ids=input_ids.long(), attention_mask=attention_mask
                            )
                            loss = criterion(out.permute(0, 2, 1).float(), labels.long())

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    else:
                        out = self.forward(input_ids=input_ids.long())
                        loss = criterion(out.permute(0, 2, 1).float(), labels.long())
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

                train_loss = total_loss / batch_count
                results[str(epoch + 1)]["train"]["loss"] = train_loss
                results[str(epoch + 1)]["eval"] = eval_results
                results[str(epoch + 1)]["eval"]["loss"] = eval_loss
                write_results(results=results, output_dir=output_dir)

                if callback != None:
                    callback.save(
                        model=self,
                        results=results,
                        path=output_dir,
                        name="model",
                        indicator_name=str(epoch),
                    )

                if scheduler != None:
                    scheduler.step()

                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}")
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Evaluation Loss: {eval_loss}")

        return dict(results[str(num_epochs)])

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
            total=total,
            desc="Evaluating",
            disable=xtagger.DISABLE_PROGRESS_BAR,
            position=0,
            leave=False,
        ) as progressbar:
            for idx, batch in enumerate(dev_dataloader):
                self.eval()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                out = self.forward(input_ids=input_ids.long(), attention_mask=attention_mask)
                loss = criterion(out.permute(0, 2, 1).float(), labels.long())

                preds = padded_argmax_and_flatten(out, pad_tag_id=label_encoder.pad_tag_id)
                ground_truth = labels.contiguous().view(-1)

                non_pad_indices = (ground_truth != label_encoder.pad_tag_id).nonzero()
                non_pad_preds = preds[non_pad_indices].squeeze(dim=-1).tolist()
                non_pad_ground_truth = ground_truth[non_pad_indices].squeeze(dim=-1).tolist()

                y_pred_test.extend([label_encoder[t] for t in non_pad_preds])
                y_true_test.extend([label_encoder[t] for t in non_pad_ground_truth])

                total_loss += loss.item()
                batch_count += 1

                progressbar.update()

        results = metric_results(
            y_pred=y_pred_test,
            y_true=y_true_test,
            eval_metrics=eval_metrics,
            tags=label_encoder.maps,
        )
        total_loss = total_loss / batch_count

        return results, total_loss

    def evaluate(
        self,
        test_set: Union[xtagger.DATASET_TYPE, DataLoader],
        tokenizer: TokenizerBase,
        device: torch.device,
        label_encoder: LabelEncoder,
        batch_size: int,
        max_length: int,
        criterion: Optional[nn.Module] = None,
        eval_metrics: List[BaseMetric] = [Accuracy],
        pretokenizer: Callable = xtagger.DEFAULT_PRETOKENIZER,
    ):
        self.to(device)
        if type(test_set) != DataLoader:
            test_dataloader = convert_to_dataloader(
                dataset=test_set,
                tokenizer=tokenizer,
                label_encoder=label_encoder,
                batch_size=batch_size,
                max_length=max_length,
                pretokenizer=pretokenizer,
                shuffle=False,
            )

        else:
            test_dataloader = test_set

        results = defaultdict(lambda: defaultdict(dict))
        y_true_test = []
        y_pred_test = []
        total_loss = 0
        batch_count = 1

        total = len(test_dataloader)
        with tqdm(
            total=total,
            desc="Testing",
            disable=xtagger.DISABLE_PROGRESS_BAR,
            position=0,
            leave=True,
        ) as progressbar:
            for idx, batch in enumerate(test_dataloader):
                self.eval()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                out = self.forward(input_ids=input_ids.long(), attention_mask=attention_mask)
                loss = criterion(out.permute(0, 2, 1).float(), labels.long())

                preds = padded_argmax_and_flatten(out, pad_tag_id=label_encoder.pad_tag_id)
                ground_truth = labels.contiguous().view(-1)

                non_pad_indices = (ground_truth != label_encoder.pad_tag_id).nonzero()
                non_pad_preds = preds[non_pad_indices].squeeze(dim=-1).tolist()
                non_pad_ground_truth = ground_truth[non_pad_indices].squeeze(dim=-1).tolist()

                y_pred_test.extend([label_encoder[t] for t in non_pad_preds])
                y_true_test.extend([label_encoder[t] for t in non_pad_ground_truth])

                total_loss += loss.item()
                batch_count += 1

                progressbar.update()

        test_results = metric_results(
            y_pred=y_pred_test,
            y_true=y_true_test,
            eval_metrics=eval_metrics,
            tags=label_encoder.maps,
        )
        total_loss = total_loss / batch_count

        results["test"] = test_results
        results["test"]["loss"] = total_loss

        return results

    @torch.inference_mode()
    def predict(
        self,
        sentence: Union[str, List[str]],
        tokenizer: TokenizerBase,
        label_encoder: LabelEncoder,
        device: torch.device,
        max_length: Optional[int] = None,
        pretokenizer: Callable = xtagger.DEFAULT_PRETOKENIZER,
    ):
        self.eval()
        if type(sentence) == str:
            sentence = [sentence]

        predictions = []
        with tqdm(
            total=len(sentence),
            desc="Prediction",
            disable=xtagger.DISABLE_PROGRESS_BAR,
            position=0,
            leave=True,
        ) as progressbar:
            for sequence in sentence:
                encoded = tokenizer.encode(
                    sentence=sequence, max_length=max_length, pretokenizer=pretokenizer
                )

                input_ids = to_tensor(encoded["input_ids"]).to(device)
                attention_mask = to_tensor(encoded["attention_mask"]).to(device)
                out = self.forward(input_ids=input_ids.long(), attention_mask=attention_mask)

                preds = (
                    padded_argmax_and_flatten(out, pad_tag_id=label_encoder.pad_tag_id)
                    .int()
                    .tolist()
                )
                preds = list(zip(input_ids.squeeze(dim=0).int().tolist(), preds))
                preds = list(
                    map(
                        lambda x: (
                            to_string(tokenizer.decode(x[0], remove_special_tokens=False)[0]),
                            label_encoder[x[1]],
                        ),
                        preds,
                    )
                )
                predictions.append(preds)
                progressbar.update()

        return predictions

    def _autocast(self, dtype=torch.float16):
        class AMPContextManager:
            def __enter__(self):
                self.dtype = dtype
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

            def __call__(self, func):
                def autocast_decorator(*args, **kwargs):
                    with autocast(dtype=self.dtype):
                        return func(*args, **kwargs)

                return autocast_decorator

        return AMPContextManager()
