import re
from typing import List, Tuple

import xtagger
from xtagger.models.regex.base import RegexBase


class RegexTagger(RegexBase):
    def __init__(self, rules: List[Tuple[str, str]] = xtagger.DEFAULT_REGEX_RULE) -> None:
        self.rules = rules

    def tag(self, token: str) -> int:
        found = -1
        for rule in self.rules:
            f = re.match(rule[0], token)
            if f != None:
                return rule[1]
        return found
