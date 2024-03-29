import os
from collections import Counter
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
import torch
import xtagger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from xtagger.tokenization.base import TokenizerBase
from xtagger.utils.helpers import flatten_list, load_pickle, save_pickle, to_string


class LabelEncoder:
    def __init__(self, dataset: xtagger.DATASET_TYPE, padding_tag: str = "[PAD]") -> None:
        self.dataset = dataset
        self.padding_tag = padding_tag
        self.reverse_maps = self.__fit()
        self.maps = {v: k for k, v in self.reverse_maps.items()}

        self.pad_tag_id = len(self.maps)

    def __fit(self):
        tags = [[pair[1] for pair in sample] for sample in self.dataset]
        tags = {item for sublist in tags for item in sublist}
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
        dataset: xtagger.DATASET_TYPE,
        tokenizer: TokenizerBase,
        label_encoder: LabelEncoder,
        max_length: int,
        pretokenizer: Callable = lambda x: x.split(),
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
        sentence = to_string([pair[0] for pair in sample])
        tags = [self.label_encoder[pair[1]] for pair in sample]

        encoded = self.tokenizer.encode(
            sentence=sentence, max_length=self.max_length, pretokenizer=self.pretokenizer
        )

        word_ids = encoded["word_ids"][0]
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded.get(
            "attention_mask", (torch.zeros_like(torch.Tensor(input_ids)) - 1).tolist()
        )

        labels = align_labels(
            tokenizer=self.tokenizer,
            label_encoder=self.label_encoder,
            tags=tags,
            word_ids=word_ids,
            input_ids=input_ids,
            label_all_tokens=True,
        )

        assert len(input_ids) == len(
            labels
        ), f"Mismatch between tokens and token labels {len(input_ids)} and {len(labels)}"

        return {
            "input_ids": torch.Tensor(input_ids),
            "labels": torch.Tensor(labels),
            "attention_mask": torch.Tensor(attention_mask),
        }


def align_labels(
    tokenizer: TokenizerBase,
    label_encoder: LabelEncoder,
    tags: List[int],
    word_ids: List[int],
    input_ids: List[int],
    label_all_tokens: bool = True,
) -> List[int]:
    if not tokenizer.subword:
        labels = [label_encoder.pad_tag_id]
        word_ids = Counter(word_ids)
        word_ids = dict(sorted(word_ids.items(), key=lambda x: x[0]))

        tags_repeat = [
            [tags[tidx]] * wid_freq for tidx, (wid, wid_freq) in enumerate(word_ids.items())
        ]
        tags_repeat = flatten_list(tags_repeat)
        labels.extend(tags_repeat)
        labels.append(label_encoder.pad_tag_id)

        pad_length = sum([1 for i in input_ids if i == tokenizer.pad_token_id])
        labels.extend([label_encoder.pad_tag_id for _ in range(pad_length)])
    else:
        previous_word_idx = None
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(label_encoder.pad_tag_id)
            elif word_idx != previous_word_idx:
                idx = tags[word_idx]
                labels.append(idx)
            else:
                idx = tags[word_idx]
                labels.append(idx if label_all_tokens else label_encoder.pad_tag_id)
            previous_word_idx = word_idx

    return labels


def convert_from_dataframe(df: pd.DataFrame) -> xtagger.DATASET_TYPE:
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
) -> xtagger.DATASET_TYPE:
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
    dataset: xtagger.DATASET_TYPE,
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
        pretokenizer=pretokenizer,
    )

    dataloader = DataLoader(sampler, batch_size=batch_size, shuffle=shuffle)
    return dataloader
