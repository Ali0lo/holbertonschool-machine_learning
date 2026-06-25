# Recurrent Neural Networks (RNN) — NumPy Implementation

## Dependencies

- Python 3
- NumPy only

---

## Files

| File | Description |
|------|-------------|
| `0-rnn_cell.py` | Single vanilla RNN cell |
| `1-rnn.py` | Full forward pass over time for RNN |
| `2-gru_cell.py` | Gated Recurrent Unit cell |
| `3-lstm_cell.py` | Long Short-Term Memory cell |
| `4-deep_rnn.py` | Multi-layer (deep) RNN forward pass |

---

## RNNCell (`0-rnn_cell.py`)

### Architecture

A single vanilla RNN cell with tanh activation and softmax output.

### Attributes

| Attribute | Shape |
|-----------|-------|
| `Wh` | `(i + h, h)` |
| `Wy` | `(h, o)` |
| `bh` | `(1, h)` |
| `by` | `(1, o)` |

### Forward Pass
"
'
