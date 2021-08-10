import numpy as np
import random
from tqdm.auto import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import xtagger
from xtagger.hmm_tagger.viterbi import Viterbi
from xtagger.hmm_tagger.hmm_utils import (
    get_emission,
    get_transition,
    get_transition_2,
    deleted_interpolation
)

from xtagger.utils.regex import (
    check_prior_tags,
    check_morphological_tags,
    EnglishRegExTagger
)

from xtagger.utils import metrics

class HiddenMarkovModel():
    def __init__(
            self,
            extend_to: str = "bigram",
            language: str = "en",
            morphological: Optional[EnglishRegExTagger] = None,
            prior: Optional[EnglishRegExTagger] = None
    ) -> None:
        r"""
        Args:
            extend_to: The type of the Hidden Markov Model. 
                       Currently supported ``bigram``, ``trigram`` and ```deleted_interpolation``.
            
            language: The language of the model. Passing this argument is not necessary now.
                      But will be necessary in future releases.

            morphological: The morphological support object from ``xtagger.utils.regex.EnglishRegExTagger``.

            prior: The prior support object from ``xtagger.utils.regex.EnglishRegExTagger``.

        """
        
        self._extend_to = extend_to
        self._morphological = morphological
        self._prior = prior
        self._extended = ["bigram", "trigram","deleted_interpolation"]
        if self._extend_to not in self._extended:
            raise ValueError("Higher than trigrams are not currently supported. Do you want to contribute?")

    def fit(self, train_set: List[List[Tuple[str, str]]], start_token: str = ".") -> None:
        r"""
        Args:
            train_set: xtagger dataset that is expresend in docs.
   
            start_token: The padding token. This is necessary since each state
                         uses previous observation and start state does not have
                         previous observation. So you have to choose a starting
                         token for calculating probabilities complete.
        """
        # multilinguality is not supported yet, but it is best practice to pass language.
        # start_token is neccesary for calculating probabilities on padded tokens
        self._train_set = train_set
        self._train_tagged_words = [tup for sent in self._train_set for tup in sent]
        self._tags = {tag for word,tag in self._train_tagged_words}
        self._vocab = {word for word,tag in self._train_tagged_words}
        self._indexing = list(self._tags)
        self._start_token = start_token

        check_morphological_tags(self._morphological, self._tags)
        check_prior_tags(self._prior, self._tags)

        if start_token not in self._tags:
            raise ValueError(f"Unknown start token: {start_token}")
        
        # fit method actually calculates transition probabilities
        # so it is normal to take shorter computational time
        # trigram and deleted_interpolation takes much more time than bigram extension
        
        if self._extend_to == "bigram":
            self._tag2tag_matrix = np.zeros((len(self._tags),len(self._tags)), dtype='float32')
            for i, tag1 in enumerate(tqdm(list(self._tags))):
                for j, tag2 in enumerate(list(self._tags)):
                    p_t1t2, pt1 = get_transition(tag1, tag2, self._train_tagged_words) #tag2 tag1
                    self._tag2tag_matrix[i, j] = p_t1t2/pt1

        elif self._extend_to == "trigram":
            self._tag2tag_matrix = np.zeros((len(self._tags),len(self._tags), len(self._tags)), dtype='float32')

            with tqdm(total =len(self._tags)**3) as t:
                for i, tag1 in enumerate(list(self._tags)):
                    for j, tag2 in enumerate(list(self._tags)):
                        for k, tag3 in enumerate(list(self._tags)):
                            p_t1t2t3, p_t1t2 = get_transition_2(tag1, tag2, tag3, self._train_tagged_words)
                            try:
                                self._tag2tag_matrix[i, j, k] = p_t1t2t3/p_t1t2
                            except:
                                self._tag2tag_matrix[i, j, k] = 0
                            t.update()

        elif self._extend_to == "deleted_interpolation":
            # deleted interpolation is proposed in Jelinek and Mercer, 1980. defined as:
            # p(t_i | t_{i-1}, t_{i-2}) = λ_1 * [C(t_{i-2}, t_{i-1}, t_i) / C(t_{i-2}, t_{i-1})] + \
            #   λ_2 * [C(t_{i-1}, t_i) / C(t_{i-1})] + λ_3 * [C(t_i) / N]
            # with constraint  λ_1 +  λ_2 +  λ_3 = 1

            # lambdas are initialized λ_1 , λ_2 , λ_3 = 0
            # for each trigram C(t1, t2, t3) > 0:
            #   i = argmax([(C(t1, t2, t3)-1)/(C(t1, t2)-1)], [(C(t2, t3)-1)/(C(t2)-1)], [(C(t3)-1)/(N-1)])
            #   λ_i += C(t1, t2, t3)
            # end for


            with tqdm(total =2*len(self._tags)**3 ) as t:
                lambdas = deleted_interpolation(self._tags, self._train_tagged_words, t)
                self._tag2tag_matrix = np.zeros((len(self._tags),len(self._tags), len(self._tags)), dtype='float32')

                for i, tag1 in enumerate(list(self._tags)):
                    for j, tag2 in enumerate(list(self._tags)):
                        for k, tag3 in enumerate(list(self._tags)):
                            N = len(self._train_tagged_words)
                            unigram = len([tup[1] for tup in self._train_tagged_words if tup[1] == tag3]) / N

                            try:
                                p_t2t3, p_t2 = get_transition(tag2, tag3, self._train_tagged_words)
                                bigram = p_t2t3 / p_t2
                            except:
                                bigram = 0

                            try:
                                p_t1t2t3, p_t1t2 = get_transition_2(tag1, tag2, tag3, self._train_tagged_words)
                                trigram = p_t1t2t3/p_t1t2
                            except:
                                trigram = 0

                            self._tag2tag_matrix[i, j, k] = lambdas[0] * unigram + lambdas[1] * bigram + lambdas[2] * trigram
                            t.update()
            print(f"λ1: {lambdas[0]}, λ2: {lambdas[1]}, λ3: {lambdas[2]}")


    def evaluate(
            self,
            test_set: List[List[Tuple[str, str]]],
            random_size: int = 30,
            seed: Optional[int] = None,
            eval_metrics: List[str] = ["acc"],
            result_type: str = "%",
            morphological: bool = True,
            prior: bool = True
    ) -> dict:

        r"""
        Args:
            test_set: xtagger dataset that is expresend in docs.
            
            random_size: Select random samples in evaluation for efficiency.

            seed: Random seed.

            eval_metrics: Current implemented eval metrics as string. Or a custom 
                          metric that is inherited from ``xtagger.utils.metrics.xMetric``
                          See docs for how to write custom metrics.

            result_type: pass ``%`` for percentage, else decimal numbers.

            morphological: if true, uses morphological object that is 
                           initialized at ``__init__``

            prior: if true, uses prior object that is initialized at ``__init__``.

        Returns:
           Dictionary of dictionaries, or dictionary with ints with metric results.
        """

        # Evaluation on full test set takes soooooo long
        # because it calls viterbi decoder with O(n^2) with bigram extension
        # O(n^3) with trigram extension

        # take uniformly distributed 30 test sample with random_size or wait

        self._test_set = test_set
        self._test_tagged_words = [tup for sent in self._test_set for tup in sent]
        self._metrics = eval_metrics

        metrics.check_eval_metrics(self._metrics)

        if seed != None:
            random.seed(seed)

        if random_size != -1:
            random_sample_indices = [random.randint(1,len(self._test_set)-1) for x in range(random_size)]
            self._eval_indices = random_sample_indices
            test_subsamples = [self._test_set[i] for i in random_sample_indices]
        else:
            test_subsamples = [self._test_set[i] for i in range(len(self._test_set))]

        test_run_base = [tup for sent in test_subsamples for tup in sent]
        test_tagged_words = [tup[0] for sent in test_subsamples for tup in sent]


        viterbi_object = Viterbi(
            words = test_tagged_words,
            tag2tag_matrix = self._tag2tag_matrix,
            train_set = self._train_tagged_words,
            extend_to = self._extend_to,
            start = self._start_token,
            morphological = self._morphological if morphological==True else None,
            prior = self._prior if prior==True else None,
            indexing = self._indexing
        )

        if self._extend_to == "bigram":
            tagged_seq = viterbi_object.fit_bigram()

        else:
            tagged_seq = viterbi_object.fit_trigram()

        preds = [pred_pair[1] for pred_pair in tagged_seq]
        ground_truth = [gt_pair[1] for gt_pair in test_run_base]
        preds_onehot, gt_onehot = metrics.tag2onehot(preds, ground_truth, self._indexing)

        results = metrics.metric_results(gt_onehot, preds_onehot, eval_metrics, result_type, self._indexing)
        return results

    def predict(
            self,
            words: List[str],
            morphological: bool = False,
            prior: bool = False
    ) -> List[Tuple[str, str]]:
        r"""
        Args:
            words: List of words.

            morphological: if true, uses morphological object that is 
                           initialized at ``__init__``

            prior: if true, uses prior object that is initialized at ``__init__``.

        Returns:
            List of tagged words (tuples): x-tagger dataset.
        """
        
        viterbi_object = Viterbi(
            words = words,
            tag2tag_matrix = self._tag2tag_matrix,
            train_set = self._train_tagged_words,
            extend_to = self._extend_to,
            start = self._start_token,
            morphological = self._morphological if morphological==True else None,
            prior = self._prior if prior==True else None,
            indexing = self._indexing
        )

        if self._extend_to == "bigram":
            tagged_seq = viterbi_object.fit_bigram()

        else:
            tagged_seq = viterbi_object.fit_trigram()
        return tagged_seq

    def set_test_set(self, test):
        self._test_set = test_set

    def __repr__(self):
        return f"HiddenMarkovModel(extend_to={self._extend_to},start_token={self._start_token}, language={self.language})"+\
            "\nTags: {self._tags}"

    def get_tags(self):
        return self._tags

    def get_start_token(self):
        return self._start_token

    def get_extension(self):
        return self._extend_to

    def get_transition_matrix(self):
        return self._tag2tag_matrix

    def get_eval_metrics(self):
        return self._metrics

    def get_metric_onehot_indexing(self):
        return self._indexing

    def eval_indices(self):
        if hasattr(self, "_eval_indices"):
            return self._eval_indices
        else:
            return -1
