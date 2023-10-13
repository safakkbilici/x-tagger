import os
import pickle
import sys
from typing import Any, List, Optional, Union

import torch


def readfile(path: str, lines: bool = False) -> Union[str, List[str]]:
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            data = f.readlines()
        else:
            data = f.read()

    return data


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def makepath(*path) -> str:
    return os.path.join(*path)


def flatten_list(ls: List) -> List:
    return [item for sublist in ls for item in sublist]


def to_tensor(ls: List) -> torch.Tensor:
    return torch.Tensor(ls)


def to_string(ls: str, sep: str = " ") -> str:
    return sep.join(ls)


def is_none(obj: Any) -> bool:
    return obj is None


def suppres_print(f):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, "w")
        value = f(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return value

    return wrapper


def padded_argmax_and_flatten(logits: torch.Tensor, pad_tag_id: int) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    _, topk_indices = torch.topk(probabilities, 2, dim=-1)
    max_prob_indices = (
        torch.where(
            topk_indices[:, :, 0] == pad_tag_id, topk_indices[:, :, 1], topk_indices[:, :, 0]
        )
        .squeeze(dim=0)
        .flatten()
    )
    assert (
        pad_tag_id not in max_prob_indices
    ), "If this happens, you think you have solved the bug but you are wrong."
    return max_prob_indices
