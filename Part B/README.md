Tamil Transliteration Seq2Seq with Attention
This repository implements a sequence-to-sequence (Seq2Seq) model with attention for Tamil transliteration tasks. It includes data loading, training, and attention visualization utilities.

Project Structure
bash
Copy
Edit
project/
├── attention.py         # Encoder and Attention Decoder implementations
├── seq2seq_attn.py      # Seq2SeqAttn model definition (Encoder + Attention Decoder)
├── data.py              # Dataset loading, vocab creation, collate_fn for batching
├── heatmap.py           # Utilities for visualizing attention heatmaps
├── train.py             # Training script with argument parsing and wandb integration
Requirements
Python 3.7+

PyTorch

wandb (Weights & Biases) for experiment tracking

Optional: torchtext or any other text preprocessing tools if customizing vocab

Install dependencies with:

bash
Copy
Edit
pip install torch wandb
Dataset
The dataset should be prepared as pairs of source-target transliteration strings.

Place the dataset in a folder and specify the path via --data_path argument.

data.py contains functions to load and preprocess the dataset into training and validation pairs.

Usage
Train the Model
bash
Copy
Edit
python train.py --data_path ./data \
                --embed_size 256 \
                --hidden_size 256 \
                --num_layers 3 \
                --dropout 0.3 \
                --cell_type LSTM \
                --batch_size 32 \
                --lr 0.001 \
                --epochs 10
Arguments
Argument	Description	Default
--data_path	Path to dataset folder	.
--embed_size	Embedding dimension size	256
--hidden_size	Hidden state size of RNN/LSTM/GRU	256
--num_layers	Number of RNN layers	3
--dropout	Dropout probability	0.3
--cell_type	Type of RNN cell: RNN, LSTM, or GRU	LSTM
--batch_size	Batch size	32
--lr	Learning rate	0.001
--epochs	Number of training epochs	10

Features
Custom Vocabulary: Builds vocabularies from dataset characters with special tokens (<pad>, <sos>, <eos>, <unk>).

Seq2Seq with Attention: Implements encoder-decoder with attention mechanism for improved transliteration accuracy.

DataLoader with Padding: Handles variable-length sequences with custom collate function and padding.

Training with wandb: Logs training loss and hyperparameters for easy experiment tracking.

Attention Heatmaps: Utilities available for visualizing attention weights during inference (heatmap.py).

How it works
The script loads train/dev data pairs.

Vocabularies for source and target are constructed from the training data.

DataLoader batches are padded and collated dynamically.

The Seq2SeqAttn model is initialized and trained using cross-entropy loss ignoring padding.

Training progress and loss are logged to Weights & Biases.

Next steps
Implement evaluation and inference scripts.

Add beam search decoding for better predictions.

Use pretrained embeddings if applicable.

Add support for more datasets and language pairs.

