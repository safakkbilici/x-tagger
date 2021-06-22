import numpy as np
import random
import time
from tqdm import tqdm
from xtagger.hmm_tagger.viterbi import Viterbi
from xtagger.hmm_tagger.hmm_utils import get_emission, \
    get_transition, get_transition_2, deleted_interpolation

class HiddenMarkovModel():
    def __init__(self, train_set, test_set, extend_to = "bigram", start_token = ".", language="en",
                 tags= ["NUM","CONJ","X","ADJ","DET","VERB","NOUN","PRT","ADV",".","ADP","PRON"]):

        if start_token not in tags:
            raise ValueError(f"Unknown start token: {start_token}")

        if len(set(tags)) != len(tags):
            raise ValueError("Duplicate tokens.")

        # multilinguality is not supported yet, but it is best practice to pass language.
        # extend_to is derivation of HMM model
        # we take train and test sets while initializing model but this can be changed by set_test_set()
        # start_token is neccesary for calculating probabilities on padded tokens

        # normally, tags in training set is automatically detected
        # the tags parameter at initialization will be removed, until removal, pass it

        
        self._language = language
        self._extend_to = extend_to
        self._train_set = train_set
        self._test_set = test_set
        self._start_token = start_token
        self._extended = ["bigram", "trigram","deleted_interpolation"]
        if self._extend_to not in self._extended:
            raise ValueError("Higher than trigrams are not currently supported. Would you want to contribute?")

        self._train_tagged_words = [tup for sent in self._train_set for tup in sent]
        self._test_tagged_words = [tup for sent in self._test_set for tup in sent]

        self._tags = {tag for word,tag in self._train_tagged_words}
        self._vocab = {word for word,tag in self._train_tagged_words}

        self._indexing = list(self._tags)

        if set(self._tags) != set(tags):
            raise ValueError("Tokens does not matching with training data.")

        if start_token not in self._tags: #not necessary but ok.
            raise ValueError(f"Unknown start token: {start_token}")


    def fit(self):
        # fit method actually calculates transition probabilities
        # so it is normal to take shorter computational time
        # trigram and deleted_interpolation takes much more time than bigram extension
        
        if self._extend_to == "bigram":
            self.tag2tag_matrix = np.zeros((len(self._tags),len(self._tags)), dtype='float32')
            for i, tag1 in enumerate(tqdm(list(self._tags))):
                for j, tag2 in enumerate(list(self._tags)):
                    p_t1t2, pt1 = get_transition(tag1, tag2, self._train_tagged_words) #tag2 tag1
                    self.tag2tag_matrix[i, j] = p_t1t2/pt1

        elif self._extend_to == "trigram":
            self.tag2tag_matrix = np.zeros((len(self._tags),len(self._tags), len(self._tags)), dtype='float32')

            for i, tag1 in enumerate(tqdm(list(self._tags))):
                for j, tag2 in enumerate(list(self._tags)):
                    for k, tag3 in enumerate(list(self._tags)):
                        p_t1t2t3, p_t1t2 = get_transition_2(tag1, tag2, tag3, self._train_tagged_words)
                        self.tag2tag_matrix[i, j, k] = p_t1t2t3/p_t1t2

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
            
            lambdas = deleted_interpolation(self._tags, self._train_tagged_words)
            print(lambdas)
            self.tag2tag_matrix = np.zeros((len(self._tags),len(self._tags), len(self._tags)), dtype='float32')
            for i, tag1 in enumerate(tqdm(list(self._tags))):
                for j, tag2 in enumerate(list(self._tags)):
                    for k, tag3 in enumerate(list(self._tags)):
                        N = len(self._train_tagged_words)
                        unigram = len([tup[1] for tup in self._train_tagged_words if tup[1] == tag3]) / N

                        p_t2t3, p_t2 = get_transition(tag2, tag3, self._train_tagged_words)
                        bigram = p_t2t3 / p_t2


                        p_t1t2t3, p_t1t2 = get_transition_2(tag1, tag2, tag3, self._train_tagged_words)
                        trigram = p_t1t2t3/p_t1t2

                        self.tag2tag_matrix[i, j, k] = lambdas[0] * unigram + lambdas[1] * bigram + lambdas[2] * trigram


    def evaluate(self, random_size = 30, all_test_set = False, seed = None, return_all=False):

        # Evaluation on full test set takes soooooo long
        # because it calls viterbi decoder with O(n^2) with bigram extension
        # O(n^3) with trigram extension

        # take uniformly distributed 30 test sample with random_size or wait

        if seed != None:
            random.seed(seed)

        if all_test_set == False:
            rndom = [random.randint(1,len(self._test_set)) for x in range(random_size)]
            test_run = [self._test_set[i] for i in rndom]
        else:
            test_run = [self._test_set[i] for i in range(len(self._test_set))]
        test_run_base = [tup for sent in test_run for tup in sent]
        test_tagged_words = [tup[0] for sent in test_run for tup in sent]


        viterbi_object = Viterbi(test_tagged_words, self.tag2tag_matrix, self._train_tagged_words,
                               self._extend_to, self._start_token, self._indexing)


        if self._extend_to == "bigram":
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
        viterbi_object = Viterbi(words, self.tag2tag_matrix, self._train_tagged_words,
                                 self._extend_to, self._start_token, self._indexing)

        if self._extend_to == "bigram":
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
