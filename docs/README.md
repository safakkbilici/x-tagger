# x-tagger Documentation
## Table of Contents  
- [1. x-tagger Dataset](#dataset)  
	- [1.1. A Default x-tagger Dataset](#nltk)
	- [1.2. x-tagger Dataset to ```pandas.DataFrame```](#x2p)
	- [1.3. ```pandas.DataFrame``` to x-tagger Dataset](#p2x)
	- [1.4. x-tagger Dataset to ```torchtext``` Iterator](#x2t)
	- [1.5 x-tagger Dataset to 🤗 datasets](#x2hf)
- [2. Models](#models)
	- [2.1. ```xtagger.HiddenMarkovModel```](#hmm)
	- [2.2. ```xtagger.LSTMForTagging```](#lstm)
	- [2.3. ```xtagger.BERTForTagging```](#bert)
- [3. Metrics](#metrics)

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

### ```xtagger.xtagger_dataset_to_df(dataset, row_as_list=False)```
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

### ```xtagger.df_to_xtagger_dataset(df)```
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

### ```xtagger.df_to_hf_dataset(df, tags, tokenizer, device)```
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

<a name="hmm"/>

### ```xtagger.HiddenMarkovModel(extend_to = "bigram", language="en", morphological = None, prior = None)```
- ```extend_to```: type of HiddenMarkovModel. Current implementations: \["bigram", "trigram", "deleted_interpolation\]
- ```language```: Language of model. Not important but best practice to use.
- ```morphological```: ```xtagger.[Language]RegexTagger``` object with ```mode = "morphological"``` parameter.
- ```prior```: ```xtagger.[Language]RegexTagger``` object with ```mode = "prior"``` parameter.
	* ```HiddenMarkovModel.fit(train_set, start_token = ".")```
		* ```train_set```: x-tagger Dataset for training.
		* ```start_token```: Start token in your training tags.
	* ```HiddenMarkovModel.evaluate(test_set, random_size = 30, seed = None, eval_metrics=["acc"], result_type = "%", morphological = True, prior = True)```
		*  ```test_set```: x-tagger Dataset for evaluating.
		*  ```random_size```: Select random samples in evaluation for efficiency.
		*  ```seed```: Random seed.
		*  ```eval_metrics```: Evaluation metrics. See more at ```xtagger.utils.metrics``` section.
		*  ```result_type```: For percentage "%", else decimal number.
		*  ```morphological```: For using initialized ```xtagger.[Language]RegexTagger```. Flexibility comes from passing it at initialiation but not using with ```morphological = False```.
		*  ```prior```: For using initialized ```xtagger.[Language]RegexTagger```. Flexibility comes from passing it at initialiation but not using with ```prior = False```.
	* ```HiddenMarkovModel.predict(words, morphological = False, prior = False)```
		* ```words```: List of words for your sentence.
		* ```morphological```: For using initialized ```xtagger.[Language]RegexTagger```.
		* ```prior```: For using initialized ```xtagger.[Language]RegexTagger```.
	* ```HiddenMarkovModel.set_test_set(test)```
	* ```HiddenMarkovModel.get_tags(test)```
	* ```HiddenMarkovModel.get_start_token(test)```
	* ```HiddenMarkovModel.get_extension(test)```
	* ```HiddenMarkovModel.get_transition_matrix(test)```
	* ```HiddenMarkovModel.get_eval_metrics(test)```
	* ```HiddenMarkovModel.get_metric_onehot_indexing(test)```
	* ```HiddenMarkovModel.eval_indices(test)```


_Note_: Evaluation takes much more time than fitting. This is because of complexity of viterbi decoding. ```random_size``` can give convergent results when considering law of large numbers. As a result, complexity of trigram and deleted_interpolation is higher than bigram. We will release benchmarks of x-tagger.

<a name="lstm"/>

### ```xtagger.LSTMForTagging(input_dim, output_dim, TEXT, TAGS, embedding_dim = 100, hidden_dim = 128, n_layers = 2, bidirectional = True, dropout = 0.25, cuda = True, tag_pad_idx = None, pad_idx = None)```
- ```input_dim```: Size of your vocab. Should be len(TEXT).
- ```output_dim```: Number of tags + \<pad\> token. Should be len(TAGS).
- ```TEXT```: Word vocabulary with ```torchtext.data.field.Field```.
- ```TAGS```: Tag vocabulary with ```torchtext.data.field.Field```.
- ```embedding_dim```: Embedding dimension of ```torch.nn.Embedding```.
- ```hidden_dim```: Hidden dimension of ```torch.nn.LSTM```.
- ```n_layers```: Number of layers of LSTM.
- ```bidirectional```: True for bidirectional LSTM, else false.
- ```dropout```: Dropout rate for fully connected network out.
- ```cuda```: If you have cuda but not to use it.
- ```tag_pad_idx```: Tag pad index. Should be ```TAGS.vocab.stoi[TAGS.pad_token]```.
- ```pad_idx```: Text pad index. Should be ```TEXT.vocab.stoi[TEXT.pad_token]```.
	* ```LSTMForTagging.fit(train_set, test_set, epochs = 10, eval_metrics = ["acc"], result_type = "%", checkpointing = None)```
		* ```train_set```: Training dataset with ```torchtext.data.iterator.BucketIterator```.
		* ```test_set```: Evaluation dataset ```torchtext.data.iterator.BucketIterator```.
		* ```epochs```: Number of epochs for training.
		* ```eval_metrics```: Evaluation metrics. See more for ```xtagger.utils.metrics``` section.
		* ```result_type```: For percentage "%", else decimal number.
		* ```checkpointing```: Checkpointing object. See more at ```xtagger.utils.callbacks.Checkpointing``` section.
	* ```LSTMForTagging.evaluate(test_set = None, eval_metrics = ["acc"], result_type = "%")```
		* ```test_set```: Test dataset ```torchtext.data.iterator.BucketIterator```. If None, uses ```test_set``` from initialization automatically.
		* ```eval_metrics```: Evaluation metrics. See more for ```xtagger.utils.metrics``` section.
		* ```result_type```: For percentage "%", else decimal number.
	* ```LSTMForTagging.predict(sentence)```:
		* ```sentence```: List of words or single string.
		* returns zipped version of words and tags and unk words: ```zipped, unk = model.predict("hello world")```

<a name="bert"/>

### BERT
                       
Not tested yet :( Working on new release.

<a name="metrics"/>

## ```xtagger.utils.metrics```
```xtagger.utils.metrics``` is a module that is hidden from the user. It provides metrics to user at training and evaluation. There is 8 built-in metric you can choose for ```eval_metric``` parameter of fit and initialization methods:

```
- avg_f1
- avg_precision
- avg_recall
- acc
- classwise_f1
- classwise_precision
- classwise_recall
- report
```

Metrics that start with "avg" (f1, precision, recall) provides micro, macro and weighted calculations. Metric that start with "classwise" (f1, precision, recall) returns metrics classwise. For example f1 score of "ADV" tag. "report" metric prints beautiful ```sklearn.metrics.classification_report```. For LSTM and BERT, those metrics are calculated for both train set and eval set at training. ```model.evaluation``` returns for only test set. Each score provided with dictionary.

## ```xtagger.utils.metrics.xMetrics```

User might want to calculate specific metric for the task. ```xtagger.utils.metrics.xMetrics``` handles it.

- ```y_true```: Suppose it as one-hot representation of each class.
- ```y_pred```: Suppose it as one-hot representation of each class.
- ```tags```: Not necessary but can be useful for beautiful printing.

_Example_: From scratch accuracy for HMM.

```python
from xtagger import xMetrics

class MyAcc(xMetrics):
    def __init__(self, y_true, y_pred, tags):
        super(MyAcc, self).__init__(y_true, y_pred, tags)
        
    def __call__(self):
        import numpy as np
        acc = 0
        for gt, pred in zip(self.y_true, self.y_pred):
            gt_index   = np.where(gt == 1)[0].item()
            pred_index = np.where(pred == 1)[0].item()
            
            if gt_index == pred_index:
                acc += 1
                
        return acc / self.y_true.shape[0]
	
model = HiddenMarkovModel(
    extend_to = "bigram",
    language = "en"
)

model.fit(train_set)

model.evaluate(
    test_set,
    random_size = -1,
    seed = 15,
    eval_metrics = ['acc', 'avg_f1', MyAcc],
    result_type = "%",
    morphological = True,
    prior = True
)
```

