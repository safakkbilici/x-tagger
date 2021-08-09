import re
import xtagger
from typing import List, Optional, Tuple, Union

def check_prior_tags(prior, model_tags):
    if prior == None:
        return
    elif type(prior) not in xtagger.IMPLEMENTED_REGEX_LANGUAGES:
        raise TypeError(f"The tagger must be [Language]RegExTagger, current implemented languages: {xtagger.IMPLEMENTED_REGEX_LANGUAGES}")
    else:
        prior_tags = [pair[1] for pair in prior.get_patterns()]
        for tag in prior_tags:
            if tag not in model_tags:
                raise ValueError("Passing different tags from training set is ambigious.")

def check_morphological_tags(morphological, model_tags):
    if morphological == None:
        return
    elif type(morphological) not in xtagger.IMPLEMENTED_REGEX_LANGUAGES:
        raise TypeError(f"The tagger must be [Language]RegExTagger, current implemented languages: {xtagger.IMPLEMENTED_REGEX_LANGUAGES}")
    else:
        morphological_tags = [pair[1] for pair in morphological.get_patterns()]
        for tag in morphological_tags:
            if tag not in model_tags:
                raise ValueError("Passing different tags from training set is ambigious.")
        

class EnglishRegExTagger(object):
    def __init__(
            self,
            rules: Optional[List[Tuple[str, str]]] = None,
            use_default: bool = True,
            mode: bool = "morphological"
    ):
        self._modes = ["morphological", "prior"]
        if mode not in self._modes:
            raise ValueError("Supporting only morphological or prior tagging for regex.")
        
        self.use_default = use_default
        self.language = "en"
        if rules is not None:
            self.patterns = []
            for pattern in rules:
                if type(pattern) != tuple:
                    raise TypeError("Matching must be tuple")
                self.patterns.append(pattern)
        if use_default:
            self.default = [
                (r'.*ing$', 'VERB'),
                (r'.*ed$', 'VERB'),
                (r'.*es$', 'VERB'),
                (r'.*\'s$', 'NOUN'),
                (r'.*s$', 'NOUN'),
                (r'\*T?\*?-[0-9]+$', 'X'),
                (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'),
                (r'.*', 'NOUN')
            ]

            if hasattr(self,"patterns") and self.patterns:
                for pattern in self.patterns:
                    if pattern not in self.default:
                        self.default.append(pattern)

    def get_patterns(self):
        if self.use_default:
            return self.default
        else:
            return self.patterns
        
    def tag(self, word):
        patterns = self.get_patterns()
        found = -1
        for rule in patterns:
            f = re.match(rule[0], word)
            if f != None:
                return rule[1]
        return found


class AutoRegExTagger(object):
    def __init__(self):
        raise NotImplementedError()
        
