import numpy as np
import random
import time
from tqdm import tqdm
from xtagger.hmm_tagger.viterbi import Viterbi
from xtagger.hmm_tagger.hmm_utils import get_emission, \
    get_transition, get_transition_2

class HiddenMarkovModel():
    def __init__(self, train_set, test_set, extend_to = "bigram", start_token = ".",
                        indexing = ["NUM","CONJ","X","ADJ","DET","VERB","NOUN","PRT","ADV",".","ADP","PRON"]):
        self.extend_to = extend_to
        self.train_set = train_set
        self.test_set = test_set
        self.indexing = indexing
        self.start_token = start_token

        if (self.extend_to != "bigram") and (self.extend_to != "trigram"):
            raise ValueError("Higher than trigrams are not currently supported. Would you want to contribute?")


    def fit(self):
        self.train_tagged_words = [tup for sent in self.train_set for tup in sent]
        self.test_tagged_words = [tup for sent in self.test_set for tup in sent]

        self.tags = {tag for word,tag in self.train_tagged_words}
        self.vocab = {word for word,tag in self.train_tagged_words}

        if self.extend_to == "bigram":
            self.tag2tag_matrix = np.zeros((len(self.tags),len(self.tags)), dtype='float32')
            for i, tag1 in enumerate(tqdm(list(self.tags))):
                for j, tag2 in enumerate(list(self.tags)):
                    p_t1t2, pt1 = get_transition(tag2, tag1, self.train_tagged_words)
                    self.tag2tag_matrix[i, j] = p_t1t2/pt1

        else:
            self.tag2tag_matrix = np.zeros((len(self.tags),len(self.tags), len(self.tags)), dtype='float32')

            for i, tag1 in enumerate(tqdm(list(self.tags))):
                for j, tag2 in enumerate(list(self.tags)):
                    for k, tag3 in enumerate(list(self.tags)):
                        p_t1t2t3, p_t1t2 = get_transition_2(tag2, tag1, tag3, self.train_tagged_words)
                        self.tag2tag_matrix[i, j, k] = p_t1t2t3/p_t1t2


    def evaluate(self, random_size = 10, all_test_set = False, seed = None, return_all=False):

        if seed != None:
            random.seed(seed)

        if all_test_set == False:
            rndom = [random.randint(1,len(self.test_set)) for x in range(random_size)]
            test_run = [self.test_set[i] for i in rndom]
        else:
            test_run = [self.test_set[i] for i in range(len(self.test_set))]
        test_run_base = [tup for sent in test_run for tup in sent]
        test_tagged_words = [tup[0] for sent in test_run for tup in sent]


        viterbi_object = Viterbi(test_tagged_words, self.tag2tag_matrix, self.train_tagged_words,
                               self.extend_to, self.start_token, self.indexing)


        if self.extend_to == "bigram":
            start = time.time()
            tagged_seq = viterbi_object.fit_bigram()
            end = time.time()

        else:
            start = time.time()
            tagged_seq = viterbi_object.fit_trigram()
            end = time.time()

        difference = end-start
        print("Time taken: {}s".format(difference))
        check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
        accuracy = len(check)/len(tagged_seq)
        print('Accuracy: {}%'.format(accuracy*100))

        if return_all == True:
            return tagged_set
        else:
            return "Done."

    def predict(self, words):
        viterbi_object = Viterbi(words, self.tag2tag_matrix, self.train_tagged_words,
                                 self.extend_to, self.start_token, self.indexing)

        if self.extend_to == "bigram":
            start = time.time()
            tagged_seq = viterbi_object.fit_bigram()
            end = time.time()

        else:
            start = time.time()
            tagged_seq = viterbi_object.fit_trigram()
            end = time.time()

        difference = end-start
        print("Time taken: {}s".format(difference))
        return tagged_seq
