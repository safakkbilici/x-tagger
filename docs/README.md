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

## NLTK Penn Treebank

NLTK Penn Treebank is most used treebank for POS tagging:

```python
import nltk
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
print(nltk_data)
```
now the ```nltk_data``` variable is in the form x-tagger dataset.

## x-tagger Dataset To ```pandas.DataFrame```

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




