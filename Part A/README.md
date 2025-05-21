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
| Hyperparameter   | Description                             | Values Swept                                 |
| ---------------- | --------------------------------------- | -------------------------------------------- |
| `embedding_size` | Size of the embedding vectors           | \[16, 32, 64, 256]                           |
| `hidden_size`    | Size of hidden layers                   | \[16, 32, 64, 256]                           |
| `enc_layers`     | Number of encoder layers                | \[1, 2, 3]                                   |
| `dec_layers`     | Number of decoder layers                | \[1, 2, 3]                                   |
| `cell_type`      | Type of recurrent cell used             | \['RNN', 'GRU', 'LSTM']                      |
| `dropout`        | Dropout rate                            | \[0.2, 0.3]                                  |
| `attention`      | Use Bahdanau attention                  | \[True, False]                               |
| `batch_size`     | Batch size used in training             | \[32, 64, 128]                               |
| `epochs`         | Number of training epochs               | 15 (constant)                                |
| `lr`             | Learning rate                           | \[1e-3, 1e-4]                                |
| `beam_size`      | Beam size for decoding (if implemented) | \[1, 3, 5] (optional / for future extension) |


- Best model will be saved as `best_model.pt`.
- Use `--attention` to enable Bahdanau attention.
- Use `--wandb` to log metrics to [Weights & Biases](https://wandb.ai/).




## Evaluation

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
