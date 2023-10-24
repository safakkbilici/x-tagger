import logging
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import xtagger
from tqdm.auto import tqdm
from xtagger.callbacks import metrics, metrics_
from xtagger.models.hmm.hmm_utils import deleted_interpolation, get_transition, get_transition_2
from xtagger.models.hmm.viterbi import Viterbi
from xtagger.models.regex.regex import RegexBase
from xtagger.utils.helpers import load_pickle, save_pickle
from xtagger.utils.validations import (
    validate_eval_metrics,
    validate_morphological_tags,
    validate_prior_tags,
)

logger = logging.getLogger(__name__)


class HiddenMarkovModel:
    def __init__(
        self,
        hmm: str = "bigram",
        morphological: Optional[RegexBase] = None,
        prior: Optional[RegexBase] = None,
    ) -> None:
        self._hmm = hmm
        self._morphological = morphological
        self._prior = prior
        self.available_hmms = ["bigram", "trigram", "deleted_interpolation"]
        if self._hmm not in self.available_hmms:
            raise NotImplementedError("Higher than trigrams are not currently supported.")

        self._train_set = None
        self._test_set = None
        self._train_tagged_words = None
        self._tags = None
        self._vocab = None
        self._indexing = None
        self._pad_token = None

    def fit(self, train_set: xtagger.DATASET_TYPE, pad_token: str = ".") -> None:
        self._train_set = train_set
        self._train_tagged_words = [tup for sent in self._train_set for tup in sent]
        self._tags = {tag for word, tag in self._train_tagged_words}
        self._vocab = {word for word, tag in self._train_tagged_words}
        self._indexing = list(self._tags)
        self._pad_token = pad_token

        validate_morphological_tags(self._morphological, self._tags)
        validate_prior_tags(self._prior, self._tags)

        if pad_token not in self._tags:
            raise ValueError(f"Invalid start token: {pad_token}")

        # fit method actually calculates transition probabilities
        # so it is normal to take shorter computational time
        # trigram and deleted_interpolation takes much more time than bigram extension

        if self._hmm == "bigram":
            self.__fit_bigram()

        elif self._hmm == "trigram":
            self.__fit_trigram()

        elif self._hmm == "deleted_interpolation":
            self.__fit_interpolation()

    def evaluate(
        self,
        test_set: xtagger.DATASET_TYPE,
        random_size: int = 30,
        seed: Optional[int] = None,
        eval_metrics: List[str | metrics_.BaseMetric] = [metrics_.Accuracy],
        morphological: bool = False,
        prior: bool = False,
    ) -> dict:
        # Evaluation on full test set takes soooooo long
        # because it calls viterbi decoder with O(n^2) with bigram extension
        # O(n^3) with trigram extension

        # take uniformly distributed 30 test sample with random_size or wait

        self._test_set = test_set

        validate_eval_metrics(eval_metrics)

        if seed != None:
            random.seed(seed)

        if random_size != -1:
            random_sample_indices = [
                random.randint(1, len(self._test_set) - 1) for _ in range(random_size)
            ]
            self._eval_indices = random_sample_indices
            test_subsamples = [self._test_set[i] for i in random_sample_indices]
        else:
            test_subsamples = [self._test_set[i] for i in range(len(self._test_set))]

        test_run_base = [tup for sent in test_subsamples for tup in sent]
        test_tagged_words = [tup[0] for sent in test_subsamples for tup in sent]

        viterbi_decoder = Viterbi(
            words=test_tagged_words,
            tag2tag_matrix=self._tag2tag_matrix,
            train_set=self._train_tagged_words,
            start=self._pad_token,
            morphological=self._morphological if morphological else None,
            prior=self._prior if prior else None,
            indexing=self._indexing,
        )

        if self._hmm == "bigram":
            tagged_seq = viterbi_decoder.decode_bigram()
        else:
            tagged_seq = viterbi_decoder.decode_trigram()

        preds = [pred_pair[1] for pred_pair in tagged_seq]
        ground_truth = [gt_pair[1] for gt_pair in test_run_base]

        results = metrics.metric_results(ground_truth, preds, eval_metrics, self._indexing)
        return results

    def predict(
        self, words: List[str], morphological: bool = False, prior: bool = False
    ) -> List[Tuple[str, str]]:
        viterbi_decoder = Viterbi(
            words=words,
            tag2tag_matrix=self._tag2tag_matrix,
            train_set=self._train_tagged_words,
            start=self._pad_token,
            morphological=self._morphological if morphological else None,
            prior=self._prior if prior else None,
            indexing=self._indexing,
        )

        if self._hmm == "bigram":
            tagged_seq = viterbi_decoder.decode_bigram()
        else:
            tagged_seq = viterbi_decoder.decode_trigram()

        return tagged_seq

    def __fit_bigram(self):
        self._tag2tag_matrix = np.zeros((len(self._tags), len(self._tags)), dtype="float32")
        with tqdm(
            total=len(self._tags) ** 2,
            desc="Fitting bigram HMM",
            disable=xtagger.DISABLE_PROGRESS_BAR,
        ) as progressbar:
            for i, tag1 in enumerate(list(self._tags)):
                for j, tag2 in enumerate(list(self._tags)):
                    p_t1t2, pt1 = get_transition(tag1, tag2, self._train_tagged_words)  # tag2 tag1
                    self._tag2tag_matrix[i, j] = p_t1t2 / pt1
                    progressbar.update()

    def __fit_trigram(self):
        self._tag2tag_matrix = np.zeros(
            (len(self._tags), len(self._tags), len(self._tags)), dtype="float32"
        )

        with tqdm(
            total=len(self._tags) ** 3,
            desc="Fitting trigram HMM",
            disable=xtagger.DISABLE_PROGRESS_BAR,
        ) as progressbar:
            for i, tag1 in enumerate(list(self._tags)):
                for j, tag2 in enumerate(list(self._tags)):
                    for k, tag3 in enumerate(list(self._tags)):
                        p_t1t2t3, p_t1t2 = get_transition_2(
                            tag1, tag2, tag3, self._train_tagged_words
                        )
                        try:
                            self._tag2tag_matrix[i, j, k] = p_t1t2t3 / p_t1t2
                        except:
                            self._tag2tag_matrix[i, j, k] = 0
                        progressbar.update()

    def __fit_interpolation(self):
        """
        deleted interpolation is proposed in Jelinek and Mercer, 1980. defined as:
        p(t_i | t_{i-1}, t_{i-2}) = λ_1 * [C(t_{i-2}, t_{i-1}, t_i) / C(t_{i-2}, t_{i-1})] + \
            λ_2 * [C(t_{i-1}, t_i) / C(t_{i-1})] + λ_3 * [C(t_i) / N]
        with constraint  λ_1 +  λ_2 +  λ_3 = 1

        lambdas are initialized λ_1 , λ_2 , λ_3 = 0
        for each trigram C(t1, t2, t3) > 0:
        i = argmax([(C(t1, t2, t3)-1)/(C(t1, t2)-1)], [(C(t2, t3)-1)/(C(t2)-1)], [(C(t3)-1)/(N-1)])
        λ_i += C(t1, t2, t3)
        end for
        """
        with tqdm(
            total=2 * len(self._tags) ** 3,
            desc="Fitting interpolated HMM",
            disable=xtagger.DISABLE_PROGRESS_BAR,
        ) as progressbar:
            lambdas = deleted_interpolation(self._tags, self._train_tagged_words, progressbar)
            self._tag2tag_matrix = np.zeros(
                (len(self._tags), len(self._tags), len(self._tags)), dtype="float32"
            )

            for i, tag1 in enumerate(list(self._tags)):
                for j, tag2 in enumerate(list(self._tags)):
                    for k, tag3 in enumerate(list(self._tags)):
                        N = len(self._train_tagged_words)
                        unigram = (
                            len([tup[1] for tup in self._train_tagged_words if tup[1] == tag3]) / N
                        )
                        try:
                            p_t2t3, p_t2 = get_transition(tag2, tag3, self._train_tagged_words)
                            bigram = p_t2t3 / p_t2
                        except:
                            bigram = 0
                        try:
                            p_t1t2t3, p_t1t2 = get_transition_2(
                                tag1, tag2, tag3, self._train_tagged_words
                            )
                            trigram = p_t1t2t3 / p_t1t2
                        except:
                            trigram = 0

                        self._tag2tag_matrix[i, j, k] = (
                            lambdas[0] * unigram + lambdas[1] * bigram + lambdas[2] * trigram
                        )
                        progressbar.update()

        logger.info(f"λ1: {lambdas[0]}, λ2: {lambdas[1]}, λ3: {lambdas[2]}")

    def save(self, path: str, name: str) -> None:
        save_pickle(self, os.path.join(path, name + ".model"))

    @staticmethod
    def load(path: str, name: str) -> "HiddenMarkovModel":
        return load_pickle(os.path.join(path, name))
