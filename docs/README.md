# x-tagger Documentation
## Table of Contents  
- [1 x-tagger Dataset](#dataset)  
	- [1.1. A Default x-tagger Dataset](#nltk)
	- [1.2. x-tagger Dataset to ```pandas.DataFrame```](#x2p)
	- [1.3. ```pandas.DataFrame``` to x-tagger Dataset](#p2x)
	- [1.4. x-tagger Dataset to ```torchtext``` Iterator](#x2t)
	- [1.5 x-tagger Dataset to 🤗 datasets](#x2hf)
- [2 Models](#models)

<a name="dataset"/>

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



```python
from xtagger import df_to_xtagger_dataset
import pandas as pd

df_train = pd.read_csv("/path/to/train.csv")
df_test = pd.read_csv("/path/to/test.csv")

data_train = df_to_xtagger_dataset(df_train)
data_test = df_to_xtagger_dataset(df_test)
```

<a name="nltk"/>

### NLTK Penn Treebank

NLTK Penn Treebank is most used treebank for POS tagging:

```python
import nltk
from sklearn.model_selection import train_test_split

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set, test_set = train_test_split(nltk_data,train_size=0.8,test_size=0.2,random_state = 2112)
```
now the ```nltk_data``` variable is in the form x-tagger dataset. In this documentation, ```train_set``` and ```test_set``` variables refer here.

<a name="x2p"/>

### x-tagger Dataset To ```pandas.DataFrame```

```xtagger.xtagger_dataset_to_df(dataset, row_as_list=False)```
- ```in_channels```: x-tagger dataset.
- ```row_as_list```: returns samples with list.

_Example_:

```python
from xtagger import xtagger_dataset_to_df

df_train = xtagger_dataset_to_df(train_set, row_as_list = True)
df_test = xtagger_dataset_to_df(test_set, row_as_list = False)
```

| sentences                       | tags        |
|---------------------------------|-------------|
| ["This", "phrase", "once", "again"]                | ["DET", "NOUN", "ADV", "ADV"]           |
| ["their", "employees", "help", "themselves"]       | ["PRON", "NOUN", "VERB", "PRON"]    |

<a name="p2x"/>

### x-tagger Dataset to ```pandas.DataFrame```

```xtagger.df_to_xtagger_dataset(df)```
- ```df```: pandas DataFrame with rows "sentence" and "tags".

<a name="x2t"/>

### x-tagger Dataset to ```torchtext.data.BucketIterator```

Most easiest way to train pytorch models comes from torchtext. You can train any token classification model that support torchtext, by converting the simples dataset x-tagger to torchtext dataset. The procedure has 2 step. First you need to transfer x-tagger dataset to ```pandas.DataFrame```. Then you can transfer ```pandas.DataFrame``` to ```torchtext.data.BucketIterator```.

```xtagger.df_to_torchtext_data(df_train, df_test, device, batch_size, pretrained_embeddings = False)```
- ```df_train```: pandas DataFrame with ```row_as_list = False```.
- ```df_test```: pandas DataFrame with ```row_as_list = False```.
- ```device```: Hardware variable ```torch.device```.
- ```batch_size```: Batch size for both df_train and df_test.
- ```pretrained_embeddings```: Default glove.6B.100d embeddings (not tested).
	* returns: three ```torchtext.data.iterator.BucketIterator``` for train, test, val and 2 ```torchtext.data.field.Field``` for TEXT and TAG vocabs.

_Example_:

```python
import torch
from xtagger import xtagger_dataset_to_df
from xtagger import df_to_torchtext_data

df_train = xtagger_dataset_to_df(train_set)
df_test = xtagger_dataset_to_df(test_set)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iterator, valid_iterator, test_iterator, TEXT, TAGS = df_to_torchtext_data(
	df_train, 
	df_test, 
	device, 
	batch_size=32
)
```


<a name="x2hf"/>

### x-tagger Dataset to 🤗 datasets

```xtagger.df_to_hf_dataset(df, tags, tokenizer, device)```
- ```df```: pandas DataFrame with ```row_as_list = True```.
- ```tags```: A list of tags in dataset.
- ```tokenizer```: An object of ```transformers.AutoTokenizer```.
- ```device```: Hardware variable ```torch.device```.

_Example_:

```python
import torch
from xtagger import xtagger_dataset_to_df
from xtagger import df_to_hf_dataset

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
<a name="models"/>

## Models

x-tagger support many algorithms! From Deep Learning to Computational Lingustics: x-tagger has LSTMs, BERTs and different types of Hidden Markov Models. Besides all, you can train any PyTorch model for pos tagging with x-tagger PyTorchTrainer wrapper! Before diving in those types and wrappers, we introduce basics of x-tagger models.

### Hidden Markov Models

You can train your **bigram** Hidden Markov Model:

```python
from xtagger import HiddenMarkovModel

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2)

hmm = HiddenMarkovModel(extend_to = "bigram", language = "en")
hmm.fit(train_set)
hmm.evaluate(test_set)
```

```xtagger.HiddenMarkovModel.evaluate()``` takes more time than ```xtagger.HiddenMarkovModel.fit()``` as expected. ```xtagger.HiddenMarkovModel.evaluate()``` evaluates 10 random datapoint from your test set without fixed seed. You can evaluate with custom n-datapoint, seed, or you can evaluate your entire test set:

```python
hmm.evaluate(test_set, seed=2112)
hmm.evaluate(test_set, random_size=30, seed=137)
hmm.evaluate(test_set, random_size=-1) #all test set
```

if you want to get tokens from  ```xtagger.HiddenMarkovModel.evaluate()```:

```python
hmm.evaluate(test_set, random_size=30, return_all=True)
```
After training you can easily get tokens for sentences:

```
hmm = HiddenMarkovModel(extend_to = "bigram", language="en")
hmm.fit(train_set)

hmm.predict(["hello","world","i","am","doing","great"])
```

```
output:
  ('hello', 'ADV'),
  ('world', 'NOUN'),
  ('i', 'ADV'),
  ('am', 'VERB'),
  ('doing', 'VERB'),
  ('great', 'ADJ')]
```

You can train your **trigram** Hidden Markov Model:

```python
import nltk
from sklearn.model_selection import train_test_split

from xtagger import HiddenMarkovModel

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2)

hmm = HiddenMarkovModel(extend_to = "trigram", language="en")
hmm.fit(train_set)
hmm.evaluate(test_set)
```

Or you can modify trigram Hidden Markov Model with **Deleted Interpolation** as proposed in Jelinek and Mercer 1980.

```python
import nltk
from sklearn.model_selection import train_test_split

from xtagger import HiddenMarkovModel

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2)

hmm = HiddenMarkovModel(extend_to = "deleted_interpolation", language="en")
hmm.fit(train_set)
hmm.evaluate(test_set, random_size=5)
```

### Long Short-Term Memory (LSTM)

You can train your **unidirectional or bidirectional** Long Short-Term Memory:

```python
import nltk
from sklearn.model_selection import train_test_split

import torch
from xtagger import LSTMForTagging
from xtagger import xtagger_dataset_to_df, df_to_torchtext_data


nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2)

df_train = xtagger_dataset_to_df(train_set)
df_test = xtagger_dataset_to_df(test_set)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iterator, valid_iterator, test_iterator, TEXT, TAGS = df_to_torchtext_data(df_train, df_test, device, batch_size=32)

model = LSTMForTagging(TEXT, TAGS, cuda=True)

model.fit(train_iterator, test_iterator)

model.predict("hello world i am doing great")
```

you can easily change your LSTM model's hyperparameters at its constructor and ```.fit()``` method.

```python
model = LSTMForTagging(TEXT, TAGS,embedding_dim=100,
		       hidden_dim=128, n_layers = 2,
                       bidirectional=True, dropout=0.25, cuda=True)
                    
model.fit(epochs=10, save_name = "lstm_save_best.pt")
model.load_best_model("lstm_save_best.pt")
```

### BERT
                       
For BERT, we use 🤗 transformers interface. You can train your BERT for token classification (using x-tagger dataset, as always):

```python
import nltk
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import torch
from xtagger import xtagger_dataset_to_df, df_to_hf_dataset
from xtagger import BERTForTagging

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
train_set,test_set =train_test_split(nltk_data,train_size=0.8,test_size=0.2)

df_train = xtagger_dataset_to_df(train_set, row_as_list=True)
df_test = xtagger_dataset_to_df(test_set, row_as_list=True)

train_tagged_words = [tup for sent in train_set for tup in sent]
tags = {tag for word,tag in train_tagged_words}
tags = list(tags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset_train = df_to_hf_dataset(df_train, tags, tokenizer, device)
dataset_test = df_to_hf_dataset(df_test, tags, tokenizer, device)

from xtagger import BERTForTagging
model = BERTForTagging("bert-base-uncased", device, tags, tokenizer)

model.fit(dataset_train, dataset_test)
model.evaluate()

preds, ids = model.predict('the next Charlie Parker would never be discouraged.')
print(preds)
````

you can easily change your BERT model's hyperparameters at its constructor and ```.fit()``` method.

```python
model = BERTForTagging("bert-base-uncased", device, tags, tokenizer,
      		       cuda=True, learning_rate = 2e-5, train_batch_size=4,
		       eval_batch_size=4, epochs=3, weight_decay=0.1)
```


                      




