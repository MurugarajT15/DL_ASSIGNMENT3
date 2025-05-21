# Tamil Transliteration — Seq2Seq with Attention

This repository contains an end‑to‑end **sequence‑to‑sequence (Seq2Seq)** model with optional Bahdanau attention for **Tamil transliteration** (romanised ⇄ Tamil script). It ships with utilities for data loading, training, and attention‑weight visualisation.

---

##  Project Structure
```text
project/
├── attention.py       # BahdanauAttention & Attention‑Decoder modules
├── seq2seq_attn.py    # Seq2SeqAttn model (Encoder + Attention Decoder)
├── data.py            # Dataset & Char‑level vocab, collate_fn
├── heatmap.py         # Plotting utilities for attention heat‑maps
├── train.py           # Training script (argparse + W&B logging)
└── README.md          # You are here
```

##  Requirements
* Python 3.7+
* [PyTorch](https://pytorch.org/)
* [Weights & Biases (wandb)](https://wandb.ai/) (optional but recommended)

Install the minimal dependencies:
```bash
pip install torch wandb
```

---

##  Dataset
*Prepare your data as **source–target** transliteration pairs (tab‑ or comma‑separated).*  
`data.py` contains helper functions that:
1. **Load** the train/val/test splits.  
2. **Build** character‑level vocabularies with specials: `<pad>`, `<sos>`, `<eos>`, `<unk>`.
3. Return a ready‑to‑use **PyTorch Dataset**.

Pass the dataset folder to the scripts via `--data_path`.

---

##  Training
```bash
python train.py \
  --data_path ./data \
  --embed_size 256 \
  --hidden_size 256 \
  --num_layers 3 \
  --dropout 0.3 \
  --cell_type LSTM \
  --batch_size 32 \
  --lr 0.001 \
  --epochs 10 \
  --attention \
  --wandb
```

###  Hyper‑parameters
| Flag | Description | Default |
|------|-------------|---------|
| `--data_path` | Path to the dataset folder | `./data` |
| `--embed_size` | Embedding dimension for source & target | `256` |
| `--hidden_size` | Hidden state size in (bi‑)RNN layers | `256` |
| `--num_layers` | # of layers in encoder **and** decoder | `3` |
| `--dropout` | Dropout probability | `0.3` |
| `--cell_type` | RNN variant: `RNN`, `LSTM`, `GRU` | `LSTM` |
| `--batch_size` | Mini‑batch size | `32` |
| `--lr` | Learning rate (Adam) | `1e‑3` |
| `--epochs` | Training epochs | `10` |
| `--attention` | Enable Bahdanau attention | *(flag)* |
| `--wandb` | Log metrics to W&B | *(flag)* |

> **Checkpoint:** The best model (lowest val loss) is saved as `best_model.pt`.

---

##  Attention Visualisation
`heatmap.py` lets you inspect where the model attends when generating each Tamil character. Example:
```python
from heatmap import plot_attention
plot_attention(src_sentence, predicted, attn_weights)
```

---

##  Features
* **Custom Char‑Vocab** with `<pad>/<sos>/<eos>/<unk>` tokens.
* **Seq2Seq + Attention**: encoder–decoder with additive (Bahdanau) attention.
* **Dynamic Batching**: custom `collate_fn` pads variable‑length sequences.
* **W&B Integration**: automatic hyper‑parameter & metric logging.
* **Attention Heat‑maps** for interpretability.

---

##  How It Works
1. **Load data** → build vocab → create `DataLoader` (with padding).
2. **Initialise** `Seq2SeqAttn` (encoder + decoder (+ attention)).
3. **Train** with cross‑entropy (ignoring `<pad>`). Teacher‑forcing ratio is configurable.
4. **Log** loss & samples to W&B; save best checkpoint.

---

##  Next Steps
-  Implement **evaluation & inference** scripts.
-  Add **beam‑search decoding** for higher accuracy.
-  Leverage **pre‑trained embeddings** (if available).
-  Extend support to additional datasets & language pairs.

---

