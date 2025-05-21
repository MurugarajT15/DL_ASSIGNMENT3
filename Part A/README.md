# Tamil Transliteration using Seq2Seq with Attention

This project implements a sequence-to-sequence (Seq2Seq) model with optional Bahdanau attention for transliterating Tamil from romanized script to native script.

##  Project Structure

```
├── data.py              # Dataset and vocabulary handling
├── model.py             # Encoder, Decoder, and Attention
├── seq2seq.py           # Seq2Seq class
├── train.py             # Training script
├── evaluate.py          # Evaluation scrip
```

##  Requirements

- Python 3.7+
- PyTorch
- wandb (optional, for logging)

Install requirements:
```bash
pip install torch wandb
```

##  Training

Run training with desired hyperparameters:

```bash
python train.py \
  --train_path path/to/ta.translit.sampled.train.tsv \
  --dev_path path/to/ta.translit.sampled.dev.tsv \
  --test_path path/to/ta.translit.sampled.test.tsv \
  --embedding_size 64 \
  --hidden_size 128 \
  --enc_layers 1 \
  --dec_layers 1 \
  --cell_type GRU \
  --dropout 0.3 \
  --batch_size 64 \
  --epochs 15 \
  --lr 1e-3 \
  --attention \
  --wandb
```

- Best model will be saved as `best_model.pt`.
- Use `--attention` to enable Bahdanau attention.
- Use `--wandb` to log metrics to [Weights & Biases](https://wandb.ai/).

##  Evaluation

Evaluate the trained model:

```bash
python evaluate.py \
  --test_path path/to/ta.translit.sampled.test.tsv \
  --model_path best_model.pt \
  --embedding_size 64 \
  --hidden_size 128 \
  --enc_layers 1 \
  --dec_layers 1 \
  --cell_type GRU \
  --dropout 0.3 \
  --attention
```

This will print word-level test accuracy.

##  Features
- Configurable encoder-decoder layers and hidden dimensions
- RNN/GRU/LSTM support
- Optional attention mechanism
- Teacher forcing for training
- Greedy decoding for evaluation

---

Feel free to add beam search, checkpointing, or inference scripts based on your needs.

##  Contact
For questions or collaboration, feel free to reach out!
