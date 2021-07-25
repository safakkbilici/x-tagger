from xtagger.hmm_tagger.hmm import HiddenMarkovModel
from xtagger.lstm_tagger.lstm_tagger import LSTMForTagging
from xtagger.bert_tagger.bert import BERTForTagging
from xtagger.utils.data_utils import df_to_xtagger_dataset
from xtagger.utils.data_utils import save_as_pickle
from xtagger.utils.data_utils import xtagger_dataset_to_df
from xtagger.utils.data_utils import df_to_torchtext_data
from xtagger.utils.data_utils import df_to_hf_dataset
from xtagger.utils.data_utils import text_to_xtagger_dataset
from xtagger.utils.regex import EnglishRegExTagger
from xtagger.utils.metrics import xMetrics



IMPLEMENTED_REGEX_LANGUAGES = [
    EnglishRegExTagger,
]

IMPLEMENTED_METRICS = [
    "avg_f1",
    "avg_precision",
    "avg_recall",
    "acc",
    "classwise_f1",
    "classwise_precision",
    "classwise_recall",
    "report",
    xMetrics
]
