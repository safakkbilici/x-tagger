from typing import List, Tuple

__version__ = "0.1.5"
__author__ = "Safak Bilici"
__license__ = "MIT"

DISABLE_PROGRESS_BAR = False
GLOBAL_INDENT = 4
DATASET_TYPE = List[List[Tuple[str, str]]]
DEFAULT_PRETOKENIZER = lambda x: x.lower().split()
DEFAULT_REGEX_RULE = [
    (r".*ing$", "VERB"),
    (r".*ed$", "VERB"),
    (r".*es$", "VERB"),
    (r".*\'s$", "NOUN"),
    (r".*s$", "NOUN"),
    (r"\*T?\*?-[0-9]+$", "X"),
    (r"^-?[0-9]+(.[0-9]+)?$", "NUM"),
    (r".*", "NOUN"),
]

from xtagger.callbacks.metrics_ import *
from xtagger.callbacks.metrics_ import BaseMetric
from xtagger.models.hmm.hmm import HiddenMarkovModel
from xtagger.models.regex.regex import RegexTagger
from xtagger.models.rnn.rnn import RNNTagger
from xtagger.tokenization.hf_interface import HFTokenizer
from xtagger.tokenization.whitespace import WhiteSpaceTokenizer
from xtagger.utils.data import (
    LabelEncoder,
    Sampler,
    convert_from_dataframe,
    convert_from_file,
    convert_to_dataloader,
)

IMPLEMENTED_METRICS = [
    F1,
    Precision,
    Recall,
    Accuracy,
    ClasswiseF1,
    ClasswisePrecision,
    ClasswiseRecall,
    ClassificationReport,
    BaseMetric,
]
