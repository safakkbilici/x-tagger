import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle

def get_emission(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)

def get_transition(tag1, tag2, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_tag1 = len([t for t in tags if t==tag1])
    count_tag1_tag2 = 0
    for idx in range(len(tags) - 1):
        if tags[idx]==tag1 and tags[idx+1] == tag2:
            count_tag1_tag2 += 1
    return count_tag1_tag2, count_tag1

def get_transition_2(tag1, tag2, tag3, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_tag1_tag2 = 0
    for idx in range(len(tags) - 1):
        if tags[idx]==tag1 and tags[idx+1] == tag2:
            count_tag1_tag2 += 1

    count_tag1_tag2_tag3 = 0
    for idx in range(len(tags) - 1):
        try:
            if tags[idx]==tag1 and tags[idx+1] == tag2 and tags[idx+3] == tag3:
                count_tag1_tag2_tag3 += 1
        except:
            pass
    return count_tag1_tag2_tag3, count_tag1_tag2


def deleted_interpolation(tags, train_tagged_words, t):
    lambdas = [0] * 3
    for i, tag1 in enumerate(list(tags)):
        for j, tag2 in enumerate(list(tags)):
            for k, tag3 in enumerate(list(tags)):
                max_list = []
                N = len(train_tagged_words)
                count_t3 = len([tup[1] for tup in train_tagged_words if tup[1] == tag3])

                count_t2 = len([tup[1] for tup in train_tagged_words if tup[1] == tag2])


                count_t2t3, count_t3 = get_transition(tag2, tag3, train_tagged_words)
                count_t1t2, count_t2 = get_transition(tag1, tag2, train_tagged_words)

                count_t1t2t3, count_t1t2 = get_transition_2(tag1, tag2, tag3, train_tagged_words)
                try:
                    max_list.append((count_t3 - 1) / (N-1))
                    max_list.append((count_t2t3 - 1) / (count_t2 - 1))
                    max_list.append((count_t1t2t3 - 1) / (count_t1t2 - 1))

                    if count_t1t2t3 > 0:
                        lmax = max(max_list)
                        lmax_idx = max_list.index(lmax)
                        lambdas[lmax_idx] += count_t1t2t3
                except:
                    pass
                t.update()

    total_sum = np.sum(lambdas)
    lambdas[0] /= total_sum
    lambdas[1] /= total_sum
    lambdas[2] /= total_sum

    return lambdas
