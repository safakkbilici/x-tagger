import os

from typing import List, Union, Callable

from xtagger.tokenization.base import TokenizerBase
from xtagger.utils.helpers import readfile, save_pickle, load_pickle


class WhiteSpaceTokenizer(TokenizerBase):
    def __init__(
        self, start_token="[START]", end_token="[END]", unk_token="[UNK]", pad_token="[PAD]"
    ) -> None:
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.special_tokens = list(vars(self).values())
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)

        self.start_token_id = self.vocab[self.start_token]
        self.end_token_id = self.vocab[self.end_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.pad_token_id = self.vocab[self.pad_token]

    def build_vocab(self):
        vocab = {t: tid for t, tid in enumerate(self.special_tokens)}
        return vocab

    def fit(self, data: str, pretokenizer: Callable = lambda x: x.split()) -> None:
        cid = self.vocab_size

        if os.path.isfile(data):
            data = readfile(data)

        data = pretokenizer(data)
        for token in data:
            if token not in self.vocab.keys():
                self.vocab[token] = cid
                cid = cid + 1

        self.i2w = {v: k for k, v in self.vocab.items()}
        print(f"Vocab size: {len(self.vocab)}")
        self.vocab_size = len(self.vocab)

    def encode(
        self,
        sentence: Union[str, List[str]],
        max_length: Optional[int],
        pretokenizer: Callable = lambda x: x.split(),
    ) -> Union[List[int], List[List[int]]]:
        sequence_length = 2
        encoded = []
        if type(sentence) == str:
            sentence = [sentence]

        for sequence in sentence:
            sequence = pretokenizer(sequence)
            encoded_sequence = []
            for token in sequence:
                tid = self.vocab.get(token, self.unk_token_id)
                encoded_sequence.append(tid)

            if max_length != None:
                if len(encoded_sequence) >= max_length:
                    encoded_sequence = encoded_sequence[: max_length - 1]
                else:
                    encoded_sequence.extend(
                        [self.pad_token_id for _ in range(len(encoded_sequence), max_len)]
                    )

            encoded_sequence.append(self.end_token_id)
            encoded.append(encoded_sequence)

    def decode(
        self, input_ids: Union[int, List[int]], remove_special_tokens: bool
    ) -> Union[str, List[str]]:
        if type(input_ids[0]) == int:
            input_ids = [input_ids]

        decoded = []
        for sequence in input_ids:
            decoded_sequence = []
            for tid in input_ids:
                decoded.append(self.i2w[tid])

            decoded.append(decoded_sequence)

        if remove_special_tokens:
            decoded = list(map(lambda x: self.remove_special_tokens(x), decoded))

        return decoded

    def remove_special_tokens(self, tokens: List[str]) -> List[str]:
        return list(lambda x: x not in self.special_tokens, filter(tokens))

    def add_tokens(self, token: Union[str, List[str]]) -> None:
        if type(token) == str:
            token = [token]

        cid = self.vocab_size
        for t in token:
            if t not in self.vocab.keys():
                self.vocab[t] = cid
                self.i2w[cid] = t

        self.vocab_size = len(self.vocab)

    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        if type(item) == str:
            return self.vocab[item]
        else:
            return self.i2w[item]

    def save(self, path: str, name: str) -> None:
        save_pickle(self, os.path.join(path, name))

    @staticmethod
    def load(path: str, name: str) -> "WhiteSpaceTokenizer":
        return load_pickle(os.path.join(path, name))
