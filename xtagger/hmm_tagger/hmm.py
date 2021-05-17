import numpy as np
import random
import time
from tqdm import tqdm
from xtagger.hmm_tagger.viterbi import Viterbi
from xtagger.hmm_tagger.hmm_utils import get_emission, \
    get_transition, get_transition_2, deleted_interpolation

class HiddenMarkovModel():
    def __init__(self, train_set, test_set, extend_to = "bigram", start_token = ".",
                        indexing = ["NUM","CONJ","X","ADJ","DET","VERB","NOUN","PRT","ADV",".","ADP","PRON"]):
        self.extend_to = extend_to
        self.train_set = train_set
        self.test_set = test_set
        self.indexing = indexing
        self.start_token = start_token
        self.extended = ["bigram", "trigram","deleted_interpolation"]
        if self.extend_to not in self.extended:
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
                    p_t1t2, pt1 = get_transition(tag1, tag2, self.train_tagged_words) #tag2 tag1
                    self.tag2tag_matrix[i, j] = p_t1t2/pt1

        elif self.extend_to == "trigram":
            self.tag2tag_matrix = np.zeros((len(self.tags),len(self.tags), len(self.tags)), dtype='float32')

            for i, tag1 in enumerate(tqdm(list(self.tags))):
                for j, tag2 in enumerate(list(self.tags)):
                    for k, tag3 in enumerate(list(self.tags)):
                        p_t1t2t3, p_t1t2 = get_transition_2(tag1, tag2, tag3, self.train_tagged_words)
                        self.tag2tag_matrix[i, j, k] = p_t1t2t3/p_t1t2

        elif self.extend_to == "deleted_interpolation":
            lambdas = deleted_interpolation(self.tags, self.train_tagged_words)
            print(lambdas)
            self.tag2tag_matrix = np.zeros((len(self.tags),len(self.tags), len(self.tags)), dtype='float32')
            for i, tag1 in enumerate(tqdm(list(self.tags))):
                for j, tag2 in enumerate(list(self.tags)):
                    for k, tag3 in enumerate(list(self.tags)):
                        N = len(self.train_tagged_words)
                        unigram = len([tup[1] for tup in self.train_tagged_words if tup[1] == tag3]) / N

                        p_t2t3, p_t2 = get_transition(tag2, tag3, self.train_tagged_words)
                        bigram = p_t2t3 / p_t2


                        p_t1t2t3, p_t1t2 = get_transition_2(tag1, tag2, tag3, self.train_tagged_words)
                        trigram = p_t1t2t3/p_t1t2

                        self.tag2tag_matrix[i, j, k] = lambdas[0] * unigram + lambdas[1] * bigram + lambdas[2] * trigram


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
