import os
import pickle

from typing import Union, List, Any


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


def makepath(*path):
    return os.path.join(*path)


def flatten_list(ls: List):
    return [item for sublist in ls for item in sublist]
