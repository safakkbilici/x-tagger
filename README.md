# x-tagger

<p align="center">
  <img src="assets/logo.png"/>
</p>

x-tagger is a Natural Language Processing toolkit for token classification in its simplest form.

* It allows you to play with all kind of data: pandas dataframe, nltk tagged corpus, .txt, torchtext iterator and 🤗 datasets.

* Supports only Hidden Markov Model with its extensions (viterbi decoding, bigram, trigram, delete interpolation), Long Short-Term Memory with its extensions (unidirectional, bidirectional), BERT; for now.

Upcoming features: Bidirectional HMMs, morphological way to dealing with unkown words in HMMs, Maximum Entropy Markov Models (MEMM), prior regex tagger.

Remainder: x-tagger is still in beta version and one-person project.

## Examples

For detailed examples, please see [docs](https://github.com/safakkbilici/x-tagger/blob/main/docs/README.md).

## Releases

Library is still-in-developement. So building from source is not recommended. Use ```pip``` for latest stable version

```bash
pip install x-tagger
```

if you are not using ```torchtext```'s legacy versioning, use:

```bash
pip install x-tagger==0.1.0
```
