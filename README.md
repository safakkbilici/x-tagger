# x-tagger

<p align="center">
  <img src="assets/logo.png"/>
</p>

x-tagger is a Natural Language Processing toolkit for token classification in its simplest form.

* It allows you to play with all kind of data: pandas dataframe, nltk tagged corpus, .txt, torchtext iterator and 🤗 datasets.

* Supports only Hidden Markov Model with its extensions (viterbi decoding, bigram, trigram, delete interpolation), Long Short-Term Memory with its extensions (unidirectional, bidirectional), BERT; for now.

* There are upcoming features soon:
  * Bidirectional Hidden Markov Models.
  * <s>Morphological way to dealing with unkown words (language dependent)</s>.
  * Maximum Entropy Markov Models (MEMM).
  * <s>Prior RegEx tagger for computational efficiency in HMMs (language dependent)</s>.
  * Beam search.
  * <s>Different metrics</s>.
  * LSTM-CNN, LSTM-CRF
  * <s>more metrics for LSTM\-\*</s>.
  * <s>saving and loading models with metric monitoring</s>.

Remainder: x-tagger is currently in beta release and one-person project.

## Examples

For detailed examples, please see [docs](https://github.com/safakkbilici/x-tagger/blob/main/docs/README.md).

## Installing

Library is still-in-developement. So building from source is not recommended. Use ```pip``` for latest stable version

```bash
pip install x-tagger
```

## Beautiful Carbon Example

<p align="center">
  <img src="assets/carbon.png"/>
</p>
