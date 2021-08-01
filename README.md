# x-tagger: A Natural Language Processing Toolkit For Sequence Labeling In Its Simplest Form.

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
  * <s> Wrapper for PyTorch sequence labeling models</s>.
  * Wrapper for PyTorch models in general.

Remainder: x-tagger is currently in beta release and one-person project.

## Getting started

### Installation

- Using pip:

```bash
pip install x-tagger
```
- Built from source:

```bash
pip install git+https://github.com/safakkbilici/x-tagger
```

### Examples

- [Hidden Markov Models](https://github.com/safakkbilici/x-tagger/blob/main/examples/Hidden%20Markov%20Models.ipynb)
- [Morphological and Prior Support](https://github.com/safakkbilici/x-tagger/blob/main/examples/Morphological%20and%20Prior%20Support.ipynb)
- [Writing Custom Metrics (Matthews Correlation Coefficient)](https://github.com/safakkbilici/x-tagger/blob/main/examples/Writing%20Custom%20Metrics%20(Matthews%20Correlation%20Coefficient).ipynb)
- [Long Short-Term Memory](https://github.com/safakkbilici/x-tagger/blob/main/examples/Long%20Short-Term%20Memory.ipynb)
- [Model Monitoring and Checkpointing](https://github.com/safakkbilici/x-tagger/blob/main/examples/Model%20Checkpointing.ipynb)
- [x-tagger PyTorchTagWrapper](https://github.com/safakkbilici/x-tagger/blob/main/examples/x-tagger%20PyTorchTagWrapper.ipynb)


## Beautiful Carbon Example

<p align="center">
  <img src="assets/carbon.png"/>
</p>
