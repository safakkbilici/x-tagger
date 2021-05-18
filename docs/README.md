# x-tagger Documentation
## x-tagger Dataset

x-tagger dataset is basically the most simples dataset for token classification.It is a list of sentences, and sentences is a list contains token/tag tuple pair:

```
[
  [('This', 'DET'),
   ('phrase', 'NOUN'),
   ('once', 'ADV'),
   ('again', 'ADV')
  ],
  [('their', 'PRON'),
   ('employees', 'NOUN'),
   ('help', 'VERB'),
   ('themselves', 'PRON')
  ]
]
```

x-tagger dataset does not have cool methods like ```.map()```, ```.build_vocab```, ```.get_batch_without_pads()```. It is jus a Python list as usual. Two questions: how can you use it for complex models, or how to get this form from custom datasets?

### NLTK Penn Treebank

NLTK Penn Treebank is most used treebank for POS tagging:

```python
import nltk
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
print(nltk_data)
```
now the ```nltk_data``` variable is in the form x-tagger dataset.

### x-tagger Dataset To ```pandas.DataFrame```

You can easily convert x-tagger dataset into ```pandas.DataFrame```, that has columns of ```["sentence", "tags"]```:

```python
import nltk
from sklearn.model_selection import train_test_split

from xtagger import xtagger_dataset_to_df

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2,random_state = 2112)

df_train = xtagger_dataset_to_df(train_set)
df_test = xtagger_dataset_to_df(test_set)
```

columns ```sentence``` and ```tags``` has form of string:

| sentences                       | tags        |
|---------------------------------|-------------|
| This phrase once again                | DET NOUN ADV ADV           |
| their employees help themselves       | PRON NOUN VERB PRON    |

if you want to get them as a list:

```python3
df_train = xtagger_dataset_to_df(train_set, row_as_list=True)
df_test = xtagger_dataset_to_df(test_set, row_as_list=True)
```

| sentences                       | tags        |
|---------------------------------|-------------|
| ["This", "phrase", "once", "again"]                | ["DET", "NOUN", "ADV", "ADV"]           |
| ["their", "employees", "help", "themselves"]       | ["PRON", "NOUN", "VERB", "PRON"]    |

### x-tagger Dataset to ```torchtext.data.BucketIterator```

Most easiest way to train pytorch models comes from torchtext. You can train any token classification model that support torchtext, by converting the simples dataset x-tagger to torchtext dataset:

```python
import nltk
from sklearn.model_selection import train_test_split
import torch

from xtagger import xtagger_dataset_to_df
from xtagger import df_to_torchtext_data

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2,random_state = 2112)

df_train = xtagger_dataset_to_df(train_set)
df_test = xtagger_dataset_to_df(test_set)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iterator, valid_iterator, test_iterator, TEXT, TAGS = df_to_torchtext_data(df_train, df_test, device, batch_size=32)
```

train, test and validation variables are ```torchtext.data.iterator.BucketIterator``` and TEXT, TAGS variables are ```torchtext.data.field.Field```

### x-tagger Dataset to 🤗 datasets

x-tagger uses 🤗 datasets for state-of-the-art models like BERT for token classification. So it is more effective to convert x-tagger dataset to 🤗 dataset format:

```python
import nltk
from sklearn.model_selection import train_test_split
import torch

from xtagger import xtagger_dataset_to_df
from xtagger import df_to_hf_dataset

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2,random_state = 2112)

df_train = xtagger_dataset_to_df(train_set, row_as_list=True)
df_test = xtagger_dataset_to_df(test_set, row_as_list=True)

train_tagged_words = [tup for sent in train_set for tup in sent]
tags = {tag for word,tag in train_tagged_words}
tags = list(tags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./path_to_tokenizer")


dataset_train = df_to_hf_dataset(df_train, tags, tokenizer, device)
dataset_test = df_to_hf_dataset(df_test, tags, tokenizer, device)
```
## Models

x-tagger supports only Hidden Markov Model with its extensions (viterbi decoding, bigram, trigram, delete interpolation), Long Short-Term Memory with its extensions (unidirectional, bidirectional), BERT; for now.

### Hidden Markov Models

You can train your bigram Hidden Markov Model:

```python
import nltk
from sklearn.model_selection import train_test_split

from xtagger import HiddenMarkovModel

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2,random_state = 2112)

hmm = HiddenMarkovModel(train_set, test_set, extend_to = "bigram")
hmm.fit()
hmm.evaluate()
```

```HiddenMarkovModel.evaluate()``` takes more time than ```HiddenMarkovModel.fit()``` as expected. ```HiddenMarkovModel.evaluate()``` evaluates 10 random datapoint from your test set without fixed seed. You can evaluate with custom n-datapoint, seed, or you can evaluate your entire test set:

```python
hmm.evaluate(seed=2112)
hmm.evaluate(random_size=30, seed=137)
hmm.evaluate(all_test_set = True)
```

if you want to get tokens from  ```HiddenMarkovModel.evaluate()```:

```python
hmm.evaluate(random_size=30, return_all=True)
```




