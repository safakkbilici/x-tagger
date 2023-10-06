__version__ = "0.1.5"
__author__ = "Safak Bilici"
__license__ = "MIT"

DISABLE_PROGRESS_BAR = False
GLOBAL_INDENT = 4

from xtagger.callbacks.metrics_ import *
from xtagger.callbacks.metrics_ import BaseMetric
from xtagger.models.hmm.hmm import HiddenMarkovModel
from xtagger.models.rnn.rnn import RNNTagger
from xtagger.utils.data import (
    Sampler,
    convert_from_dataframe,
    convert_from_file,
    convert_to_dataloader,
)
from xtagger.utils.regex import EnglishRegExTagger
from xtagger.tokenization.whitespace import WhiteSpaceTokenizer
from xtagger.utils.data import LabelEncoder

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

IMPLEMENTED_REGEX_LANGUAGES = [
    EnglishRegExTagger,
]
