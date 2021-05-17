from tqdm import tqdm
import numpy as np
from xtagger.hmm_tagger.hmm_utils import get_emission, \
    get_transition, get_transition_2

class Viterbi(object):
    def __init__(self, words, tag2tag_matrix, train_set, extend_to = "bigram", start = ".",
                 indexing = ["NUM","CONJ","X","ADJ","DET","VERB","NOUN","PRT","ADV",".","ADP","PRON"]):
        self.words = words
        self.tag2tag_matrix = tag2tag_matrix
        self.indexing = indexing
        self.extend_to = extend_to
        self.train_set = train_set
        self.start = start

        if (self.extend_to != "bigram") and (self.extend_to != "trigram"):
            raise ValueError("Higher than trigrams are not currently supported. Would you want to contribute?")

    def fit_bigram(self):
        state = []
        T = list(set([pair[1] for pair in self.train_set]))
        for key, word in enumerate(tqdm(self.words)):
            p = []
            for tag in T:
                if key == 0:
                    start_idx = self.indexing.index(self.start)
                    tag_idx = self.indexing.index(tag)
                    transition_p = self.tag2tag_matrix[start_idx,tag_idx]

                else:
                    pre_state_idx = self.indexing.index(state[-1])
                    tag_idx = self.indexing.index(tag)
                    transition_p = self.tag2tag_matrix[pre_state_idx,tag_idx]


                p_tiwi, pti = get_emission(self.words[key], tag, self.train_set)
                emission_p = p_tiwi/pti
                state_probability = emission_p * transition_p
                p.append(state_probability)

            pmax = max(p)
            state_max = T[p.index(pmax)]
            state.append(state_max)

        return list(zip(self.words, state))


    def fit_trigram(self):
        state = []
        T = list(set([pair[1] for pair in self.train_set]))
        for key, word in enumerate(tqdm(self.words)):
            p1 = []
            p2 = []
            for tag2 in T:
                for tag1 in T:
                    if key==0:
                        start_idx = self.indexing.index(self.start)
                        start2_idx = self.indexing.index(self.start)
                        tag2_idx = self.indexing.index(tag2)
                        transition_p = self.tag2tag_matrix[start_idx, start2_idx, tag2_idx]

                    elif key==1:
                        start_idx = self.indexing.index(self.start)
                        tag1_idx = self.indexing.index(tag1)
                        tag2_idx = self.indexing.index(tag2)
                        transition_p = self.tag2tag_matrix[start_idx, tag1_idx, tag2_idx]

                    else:
                        pre_state_idx = self.indexing.index(state[-1])
                        tag1_idx = self.indexing.index(tag1)
                        tag2_idx = self.indexing.index(tag2)
                        transition_p = self.tag2tag_matrix[pre_state_idx, tag1_idx, tag2_idx]

                    p_tiwi, pti = get_emission(self.words[key], tag2, self.train_set)
                    emission_p = p_tiwi/pti
                    state_probability = emission_p * transition_p
                    p1.append(state_probability)

                pmax_inner = max(p1) #max for tag2_i
                p2.append(pmax_inner)

            pmax = max(p2)
            state_max = T[p2.index(pmax)]
            state.append(state_max)

        return list(zip(self.words, state))


    def get_indexing(self):
        return self.indexing

    def get_type(self):
        return self.extend_to




