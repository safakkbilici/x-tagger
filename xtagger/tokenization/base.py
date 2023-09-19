from abc import ABC, abstractmethod
from typing import List, Union, Optional, Callable


class TokenizerBase(ABC):
    @abstractmethod
    def fit(self, corpus: str, pretokenizer: Callable) -> Union[List[int], List[List[int]]]:
        pass

    @abstractmethod
    def encode(
        self, sentence: Union[str, List[str]], max_length: Optional[int], pretokenizer: Callable
    ) -> Union[List[int], List[List[int]]]:
        pass

    @abstractmethod
    def decode(
        self, input_ids: Union[int, List[int]], remove_special_tokens: bool
    ) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def remove_special_tokens(self, input_ids: Union[int, List[int]]) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load(path: str, name: str) -> None:
        pass
