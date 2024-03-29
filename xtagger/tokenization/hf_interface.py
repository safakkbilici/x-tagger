import logging
from typing import Callable, Dict, List, Optional, Union

import xtagger
from xtagger.tokenization.base import TokenizerBase
from xtagger.utils.helpers import is_none, load_pickle, readfile, save_pickle
from xtagger.utils.logging_helpers import set_global_logging_level, suppress_hf_logs


class HFTokenizer(TokenizerBase):
    def __init__(self, name):
        from transformers import AutoTokenizer

        suppress_hf_logs()
        logging.getLogger("transformers").setLevel(logging.WARNING)

        self.subword = True
        hf_tokenizer = AutoTokenizer.from_pretrained(name)

        self.start_token = (
            hf_tokenizer.cls_token
            if not is_none(hf_tokenizer.cls_token)
            else hf_tokenizer.bos_token
        )
        self.end_token = (
            hf_tokenizer.sep_token
            if not is_none(hf_tokenizer.sep_token)
            else hf_tokenizer.eos_token
        )
        self.unk_token = hf_tokenizer.unk_token
        self.pad_token = hf_tokenizer.pad_token

        self.start_token_id = hf_tokenizer.convert_tokens_to_ids(self.start_token)
        self.end_token_id = hf_tokenizer.convert_tokens_to_ids(self.end_token)
        self.unk_token_id = hf_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.pad_token_id = hf_tokenizer.convert_tokens_to_ids(self.pad_token)

        self.vocab_size = len(hf_tokenizer)

        self.special_tokens = dict(
            start_token=self.start_token,
            end_token=self.end_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
        )
        self.hf_tokenizer = hf_tokenizer

    def encode(
        self,
        sentence: Union[List[str], List[List[str]]],
        max_length: Optional[int],
        pretokenizer: Callable = xtagger.DEFAULT_PRETOKENIZER,
    ):
        if type(sentence) == str:
            sentence = [pretokenizer(sentence)]
        else:
            sentence = [pretokenizer(s) for s in sentence]

        if max_length != None:
            encoded = self.hf_tokenizer(
                text=sentence,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                is_split_into_words=True,
            )

        else:
            encoded = self.hf_tokenizer(
                text=sentence, return_attention_mask=True, is_split_into_words=True
            )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        word_ids = encoded.word_ids()

        return {"input_ids": input_ids, "word_ids": [word_ids], "attention_mask": attention_mask}

    def remove_special_tokens(self, tokens: List[str]) -> List[str]:
        return list(filter(lambda x: x not in self.special_tokens, tokens))

    def decode(self, input_ids: Union[int, List[int]], remove_special_tokens: bool = True):
        if type(input_ids) in [int, float]:
            input_ids = [[input_ids]]

        elif type(input_ids[0]) in [int, float]:
            input_ids = [input_ids]

        decoded = []
        for sequence in input_ids:
            decoded_sequence = []
            for tid in sequence:
                decoded_sequence.append(self.hf_tokenizer.decode(tid, remove_special_tokens))
            decoded.append(decoded_sequence)
        return decoded

    def fit(*args, **kwargs):
        pass

    def load(*args, **kwargs):
        pass

    def save(*args, **kwargs):
        pass
