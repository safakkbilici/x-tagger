import pandas as pd
from tqdm import tqdm
import pickle

def df_to_hmm_dataset(df):
    data2list = []
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        ner_tags = row["ner_tags"]
        tokens = row["tokens"]
        mapped = list(map(lambda x, y: (x,y), tokens, ner_tags))
        data2list.append(mapped)
    return data2list

def save_as_pickle(hmm_dataset, name):
    with open(f"{name}.pkl", wb) as handle:
        pickle.dump(train_data2list, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


