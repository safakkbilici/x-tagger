from xtagger.hmm_tagger.hmm import HiddenMarkovModel
from xtagger.lstm_tagger.lstm_tagger import LSTMForTagging
from xtagger.bert_tagger.bert_tagger import BERTForTagging
from xtagger.utils.data_utils import (
    df_to_xtagger_dataset,
    save_as_pickle,
    xtagger_dataset_to_df,
    df_to_torchtext_data,
    df_to_hf_dataset,
    text_to_xtagger_dataset
)
    
from xtagger.utils.regex import EnglishRegExTagger
from xtagger.utils.metrics import xMetrics
from xtagger.utils.callbacks import Checkpointing
from xtagger.utils.trainer import PyTorchTagTrainer



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
    "eval_loss",
    "train_loss",
    xMetrics
]

MONITOR = [
    "train_loss",
    "train_acc",
    "train_avg_f1",
    "train_avg_precision",
    "train_avg_recall",
    "eval_loss",
    "eval_acc",
    "eval_avg_f1",
    "eval_avg_precision",
    "eval_avg_recall",
    ]
