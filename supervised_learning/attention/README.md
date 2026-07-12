# Machine Translation with RNNs

This project implements core components of a sequence-to-sequence (Seq2Seq) neural machine translation model using TensorFlow.

## Files

### `0-rnn_encoder.py`
Implements the `RNNEncoder` class.

Features:
- Embedding layer
- GRU encoder
- Hidden state initialization
- Returns encoder outputs and final hidden state

### `1-self_attention.py`
Implements the `SelfAttention` class.

Features:
- Bahdanau (Additive) Attention
- Computes attention scores
- Produces context vectors
- Returns attention weights

## Requirements

- Python 3
- TensorFlow 2.x

## Usage

Import the classes into your machine translation model:

```python
from 0-rnn_encoder import RNNEncoder
from 1-self_attention import SelfAttention

