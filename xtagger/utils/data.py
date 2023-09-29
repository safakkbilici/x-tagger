import os
import pickle
from typing import Callable, List, Tuple, Union

import pandas as pd
import torch
import xtagger
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from xtagger.tokenization.base import TokenizerBase
from xtagger.utils.helpers import load_pickle, save_pickle


class LabelEncoder:
    def __init__(self, dataset: List[List[Tuple[str, str]]], padding_tag: str = "[PAD]") -> None:
        self.dataset = dataset
        self.padding_tag = padding_tag
        self.reverse_maps = self.__fit()
        self.maps = {v: k for k, v in self.reverse_maps.items()}

        self.pad_tag_id = len(self.maps)

    def __fit(self):
        tags = [[token[0] for token in sample] for sample in self.dataset]
        tags = {item for sublist in self.dataset for item in sublist}
        reverse_maps = dict(enumerate(tags))
        return reverse_maps

    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        if type(item) == str:
            return self.maps[item]
        else:
            return self.reverse_maps[item]

    def save(self, path: str, name: str) -> None:
        save_pickle(self, os.path.join(path, name + ".label_encoder"))

    @staticmethod
    def load(path: str, name: str) -> "LabelEncoder":
        return load_pickle(os.path.join(path, name))


class Sampler(Dataset):
    def __init__(
        self,
        dataset: List[List[Tuple[str, str]]],
        tokenizer: TokenizerBase,
        label_encoder: LabelEncoder,
        max_length: int,
        pretokenizer: Callable = lambda x: x,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.pretokenizer = pretokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sentence = [pair[0] for pair in sample]
        tags = [self.label_encoder[pair[1]] for pair in sample]
        encoded = self.tokenizer.encode(
            sentence=sentence, max_length=self.max_length, pretokenizer=self.pretokenizer
        )
        labels = [item for item, count in zip(tags, encoded["word_ids"]) for _ in range(count)]

        return {"input_ids": torch.Tensor(encoded["input_ids"]), "labels": torch.Tensor(labels)}


def convert_from_dataframe(df: pd.DataFrame) -> List[List[Tuple[str, str]]]:
    data = []
    with tqdm(
        total=len(df), desc="Converting dataset", disable=xtagger.DISABLE_PROGRESS_BAR
    ) as progressbar:
        for index, row in df.iterrows():
            ner_tags = row["tags"]
            tokens = row["sentence"]
            mapped = list(map(lambda x, y: (x, y), tokens, ner_tags))
            data.append(mapped)
            progressbar.update()
    return data


def convert_from_file(
    filename: str, sep1: str = " ", sep2: str = "\t", encoding: str = "utf-8"
) -> List[List[Tuple[str, str]]]:
    total = len(open(filename, "r", encoding=encoding).readlines())
    assert sep1 != sep2, "Two separators must not be same"
    data = []
    with open(filename, "r", encoding=encoding) as f:
        for line in tqdm(
            f, total=total, desc="Converting dataset", disable=xtagger.DISABLE_PROGRESS_BAR
        ):
            line_ = line.replace("\n", "")
            line_split = line_.split(sep2)
            tokens_splitted = line_split[0].split(sep1)
            tags_splitted = line_split[1].split(sep2)

            assert len(tokens_splitted) == len(tags_splitted), "Each token must have one tag"
            mapped = list(map(lambda x, y: (x, y), tokens_splitted, tags_splitted))

            data.append(mapped)

    return data


def convert_to_dataloader(
    dataset: List[List[Tuple[str, str]]],
    tokenizer: TokenizerBase,
    label_encoder: LabelEncoder,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    pretokenizer: Callable = lambda x: x,
) -> DataLoader:
    sampler = Sampler(
        dataset=dataset,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=max_length,
        pretokenizer=pretokenizer
    )

    dataloader = DataLoader(sampler, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# def tokenize_and_align_labels(examples, tokenizer, tags, label_all_tokens):
#     tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=True)
#     labels = []
#     for i, label in enumerate(examples[f"tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:
#                 idx = tags.index(label[word_idx])
#                 label_ids.append(idx)
#             else:
#                 idx = tags.index(label[word_idx])
#                 label_ids.append(idx if label_all_tokens else -100)
#             previous_word_idx = word_idx

#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs

# def df_to_hf_dataset(df, tags, tokenizer, device, label_all_tokens=True):
#     dataset = hfd.Dataset.from_pandas(df)
#     dataset = dataset.map(
#         tokenize_and_align_labels,
#         fn_kwargs = {
#             'tokenizer': tokenizer,
#             'tags': tags,
#             'label_all_tokens': label_all_tokens
#         },
#         batched = True
#     )
#     return dataset
