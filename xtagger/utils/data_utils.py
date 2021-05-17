import pickle
import pandas as pd
import torch
from torchtext import data
from torchtext import datasets

def df_to_xtagger_dataset(df):
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

def xtagger_dataset_to_df(dataset, row_as_list=False):
    df = pd.DataFrame(columns = ["sentence", "tags"])
    for sentence in dataset:
        d = {}
        sent = []
        tag = []
        for pair in sentence:
            sent.append(pair[0])
            tag.append(pair[1])
        if not row_as_list:
            d = {"sentence": ' '.join(sent), "tags": ' '.join(tag)}
        else:
            d = {"sentence": sent, "tags": tag}
        df = df.append(d,ignore_index=True)
    return df

def df_to_torchtext_data(df_train, df_test, device, batch_size, min_freq = 2,
                         pretrained_embeddings = False):
    df_train.to_csv("train.csv",index=False)
    df_test.to_csv("test.csv",index=False)

    TEXT = data.Field(lower = True)
    TAGS = data.Field(unk_token = None)
    fields = (("sentence", TEXT), ("tags", TAGS))

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path = '.',
        train = 'train.csv',
        test = 'test.csv',
        validation = 'test.csv',
        format = 'csv',
        fields = fields,
        skip_header = True
    )
    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of testing examples: {len(test_data)}")

    for i in train_data.examples:
        if len(vars(i)["sentence"]) != len(vars(i)["tags"]):
            raise TypeError(f"Something wrong with this dataset. Tag length and sentence length mismatch at index {i}.")

    if pretrained_embeddings:
        print("Downloading glove.6B.100d...")
        TEXT.build_vocab(train_data,
                         min_freq = min_freq,
                         vectors = "glove.6B.100d",
                         #unk_init = torch.Tensor.nor
        )
    else:
        TEXT.build_vocab(train_data,
                         min_freq = min_freq,
                         #unk_init = torch.Tensor.nor
        )

    TEXT.build_vocab(train_data)
    TAGS.build_vocab(train_data)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in TAGS vocabulary: {len(TAGS.vocab)}")

    BATCH_SIZE = 32


    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size,
        sort_within_batch=False,
        sort=False,
        device = device
    )

    return train_iterator, valid_iterator, test_iterator, TEXT, TAGS

