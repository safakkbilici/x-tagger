from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union


class RegexBase(ABC):
    @abstractmethod
    def tag(self, token) -> int:
        pass
