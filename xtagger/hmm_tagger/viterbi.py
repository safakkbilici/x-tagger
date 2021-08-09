from tqdm.auto import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from xtagger.hmm_tagger.hmm_utils import (
    get_emission,
    get_transition,
    get_transition_2
)

from xtagger.utils.regex import EnglishRegExTagger

r"""
This file contails implementation of viterbi decoder at
evaluation and prediction level. We are documenting the types of
arguments and returned variables for contributors.
"""

class Viterbi(object):
    def __init__(self,
                 words: List[str],
                 tag2tag_matrix: np.ndarray,
                 train_set: List[Tuple[str, str]],
                 extend_to: str = "bigram",
                 start: str = ".",
                 morphological: Optional[EnglishRegExTagger] = None,
                 prior: Optional[EnglishRegExTagger] = None,
                 indexing: str = ["."]):
        """
        decodes maximum probabilities at evaluation and inference time
        with dynamic viterbi decoder.
        """
        
        self._words = words
        self._tag2tag_matrix = tag2tag_matrix
        self._indexing = indexing
        self._extend_to = extend_to
        self._train_set = train_set
        self._extended = ["bigram", "trigram","deleted_interpolation"]
        self._start = start
        self._morphological = morphological
        self._prior = prior

        # not neccesary
        if self._extend_to not in self._extended:
            raise ValueError("Higher than trigrams are not currently supported. Would you want to contribute?")

    # bigram HMM and trigram HMM is currently implemented extensions
    # normally, it is easy to extend it to n-gram
    # but in practice we try not to use n-grams higher than fourgram
    # the asymptotic computational of >4 is high
    def fit_bigram(self):
        state = []
        T = list(set([pair[1] for pair in self._train_set]))
        for key, word in enumerate(tqdm(self._words)):
            p = []
            
            if self._prior != None:
                state_max = self._prior.tag(word)
                if state_max != -1:
                    state.append(state_max)
                    continue
                
            for tag in T:
                if key == 0:
                    # previous word for first word is actually padding
                    # so we can set the tag as start tag
                    start_idx = self._indexing.index(self._start)
                    tag_idx = self._indexing.index(tag)
                    transition_p = self._tag2tag_matrix[start_idx,tag_idx]

                else:
                    #start with previous predicted tag
                    pre_state_idx = self._indexing.index(state[-1])
                    tag_idx = self._indexing.index(tag)
                    transition_p = self._tag2tag_matrix[pre_state_idx,tag_idx]

                # emission probabilities
                p_tiwi, pti = get_emission(self._words[key], tag, self._train_set)
                emission_p = p_tiwi/pti
                state_probability = emission_p * transition_p
                p.append(state_probability)

            pmax = max(p)
            if self._morphological != None and pmax == 0:
                state_max = self._morphological.tag(word)
                if state_max != -1:
                    state.append(state_max)
                else:
                    state_max = T[p.index(pmax)]
                    state.append(state_max)
            else:
                state_max = T[p.index(pmax)]
                state.append(state_max)

        return list(zip(self._words, state))


    def fit_trigram(self):
        state = []
        T = list(set([pair[1] for pair in self._train_set]))
        with tqdm(total = len(self._words) * len(T) * len(T)) as t:
            for key, word in enumerate(self._words):
                p1 = []
                p2 = []
                
                if self._prior != None:
                    state_max = self._prior.tag(word)
                    if state_max != -1:
                        state.append(state_max)
                        continue
                    
                for tag2 in T:
                    for tag1 in T:
                        if key==0:
                            # first two previous word for first word are actually padding
                            # so we can set the tag as start tags
                            start_idx = self._indexing.index(self._start)
                            start2_idx = self._indexing.index(self._start)
                            tag2_idx = self._indexing.index(tag2)
                            transition_p = self._tag2tag_matrix[start_idx, start2_idx, tag2_idx]

                        elif key==1:
                            # previous word for first word is actually padding
                            # so we can set the tag as start tag
                            start_idx = self._indexing.index(self._start)
                            tag1_idx = self._indexing.index(tag1)
                            tag2_idx = self._indexing.index(tag2)
                            transition_p = self._tag2tag_matrix[start_idx, tag1_idx, tag2_idx]
                            
                        else:
                            pre_state_idx = self._indexing.index(state[-1])
                            tag1_idx = self._indexing.index(tag1)
                            tag2_idx = self._indexing.index(tag2)
                            transition_p = self._tag2tag_matrix[pre_state_idx, tag1_idx, tag2_idx]
                        t.update()

                            # emission probabilities
                        p_tiwi, pti = get_emission(self._words[key], tag2, self._train_set)
                        emission_p = p_tiwi/pti
                        state_probability = emission_p * transition_p
                        p1.append(state_probability)

                    pmax_inner = max(p1) #max for tag2_i
                    p2.append(pmax_inner)

                pmax = max(p2)
                if self._morphological != None and pmax == 0:
                    state_max = self._morphological.tag(word)
                    if state_max != -1:
                        state.append(state_max)
                    else:
                        state_max = T[p.index(pmax)]
                        state.append(state_max)
                else:
                    state_max = T[p2.index(pmax)]
                    state.append(state_max)

        return list(zip(self._words, state))


    def get_indexing(self):
        return self._indexing

    def get_type(self):
        return self._extend_to

    def get_transition_matrix(self):
        return self._tag2tag_matrix
