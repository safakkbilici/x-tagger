import transformers
import datasets as hfd
import pickle
import pandas as pd
import torch

try:
    from torchtext.legacy import data
    from torchtext.legacy import datasets
except ImportError:
    from torchtext import data
    from torchtext import datasets


def df_to_xtagger_dataset(df):
    data2list = []
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        ner_tags = row["tags"]
        tokens = row["sentence"]
        mapped = list(map(lambda x, y: (x,y), tokens, ner_tags))
        data2list.append(mapped)
    return data2list

def text_to_xtagger_dataset(filename, word_tag_split = " ", word_split = "\n", sent_split="\n\n"):
    with open(filename,"r") as f:
        data = f.read()
    sentences = data.split(sent_split)
    data2list = []
    for sentence in sentences:
        word_tag_list = []
        for pair in sentence.split(word_split):
            word_tag_list.append(tuple(pair.split(word_tag_split)))
        data2list.append(word_tag_list)
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
                         #unk_init = torch.Tensor.normal_
        )
    else:
        TEXT.build_vocab(train_data,
                         min_freq = min_freq,
                         #unk_init = torch.Tensor.normal_
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

def tokenize_and_align_labels(examples, tokenizer, tags, label_all_tokens):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                idx = tags.index(label[word_idx])
                label_ids.append(idx)
            else:
                idx = tags.index(label[word_idx])
                label_ids.append(idx if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def df_to_hf_dataset(df, tags, tokenizer, device, label_all_tokens=True):
    dataset = hfd.Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_and_align_labels,
                          fn_kwargs = {'tokenizer': tokenizer,'tags': tags, 'label_all_tokens': label_all_tokens},
                          batched = True)
    return dataset
