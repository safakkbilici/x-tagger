import os

from typing import List, Union, Callable, Optional, Tuple, Dict

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
        vocab = {t: tid for tid, t in enumerate(self.special_tokens)}
        return vocab

    def fit(
        self,
        data: Union[str, List[List[Tuple[str, str]]]],
        pretokenizer: Callable = lambda x: x.split(),
    ) -> None:
        
        cid = self.vocab_size
        if os.path.isfile(data):
            data = readfile(data)
 
        elif type(data) != str:
            data = [[token[0] for token in sample] for sample in data]
            data = [item for sublist in data for item in sublist]
            data = " ".join(data)

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
        sentence: Union[List[str], List[List[str]]],
        max_length: Optional[int],
        pretokenizer: Callable = lambda x: x,
        **kwargs
    ) -> Dict[str, Union[List[int], List[List[int]]]]:
        encoded = []
        sequence_word_ids = []

        if type(sentence[0]) == str:
            sentence = [sentence]
            
        for sequence in sentence:
            encoded_sequence = []
            word_ids = []
            for wid, token in enumerate(sequence):
                token = pretokenizer(token)
                if type(token) == str: 
                    tid = self.vocab.get(token, self.unk_token_id)
                    encoded_sequence.append(tid)
                    word_ids.append(wid)
                else:
                    tid = [self.vocab.get(t, self.unk_token_id) for t in token]
                    encoded_sequence.extend(tid)
                    word_ids.extend([wid for _ in tid])

            if max_length != None:
                if len(encoded_sequence) >= max_length:
                    encoded_sequence = encoded_sequence[: max_length - 1]
                    encoded_sequence.append(self.end_token_id)
                else:
                    encoded_sequence.append(self.end_token_id)
                    encoded_sequence.extend(
                        [self.pad_token_id for _ in range(len(encoded_sequence), max_length)]
                    )

            else:
                encoded_sequence.append(self.end_token_id)

            encoded.append(encoded_sequence)
            sequence_word_ids.append(word_ids)

        return {"input_ids": encoded, "word_ids": sequence_word_ids}

    def decode(
        self, input_ids: Union[int, List[int]], remove_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        if type(input_ids[0]) == int:
            input_ids = [input_ids]

        decoded = []
        for sequence in input_ids:
            decoded_sequence = []
            for tid in sequence:
                decoded_sequence.append(self.i2w[tid])

            decoded.append(decoded_sequence)

        if remove_special_tokens:
            decoded = list(map(lambda x: self.remove_special_tokens(x), decoded))

        return decoded

    def remove_special_tokens(self, tokens: List[str]) -> List[str]:
        return list(filter(lambda x: x not in self.special_tokens, tokens))

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
        save_pickle(self, os.path.join(path, name + ".tokenizer"))

    @staticmethod
    def load(path: str, name: str) -> "WhiteSpaceTokenizer":
        return load_pickle(os.path.join(path, name))
