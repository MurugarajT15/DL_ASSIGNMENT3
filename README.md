#  Transliteration Model (Seq2Seq with Attention)

This repository implements a character-level neural sequence-to-sequence model for transliteration using PyTorch. The model is flexible and supports different RNN variants (RNN, GRU, LSTM), optional bidirectionality in the encoder, and attention in the decoder.

##  Features

- Character-level transliteration (e.g., English to Tamil)
- Encoder-decoder architecture with optional attention mechanism
- Support for RNN, GRU, or LSTM cells
- Configurable embedding size, hidden size, number of layers, dropout
- WandB integration for experiment tracking
- Easy training and extensibility

##  Model Architecture

- **Encoder:** Embedding layer + RNN (configurable)
- **Decoder:** Embedding layer + RNN with optional dot-product attention
- **Attention:** Dot-product over encoder outputs
- **Seq2Seq:** Teacher forcing during training

##  Dataset Format

The training dataset should be a text file with tab-separated input-output word pairs per line:

english_word1 tamil_word1
english_word2 tamil_word2
...

bash
Copy
Edit

## 🛠️ Installation

```bash
git clone https://github.com/your-username/transliteration-model.git
cd transliteration-model
pip install -r requirements.txt
Required packages
torch

wandb (optional for logging)

 Training
bash
Copy
Edit
python train.py \
  --data_path data/eng_tam.txt \
  --cell_type GRU \
  --embedding_size 128 \
  --hidden_size 256 \
  --n_layers 1 \
  --dropout 0.1 \
  --batch_size 32 \
  --epochs 10 \
  --lr 0.001 \
  --attention \
  --bidirectional \
  --use_wandb
Arguments
Argument	Description	Default
--data_path	Path to the input dataset	eng_ita.txt
--cell_type	RNN variant: RNN, LSTM, GRU	GRU
--embedding_size	Embedding vector size	128
--hidden_size	Hidden layer size	256
--n_layers	Number of RNN layers	1
--dropout	Dropout rate	0.1
--batch_size	Batch size	32
--epochs	Number of training epochs	10
--lr	Learning rate	0.001
--attention	Use attention mechanism	False
--bidirectional	Use bidirectional encoder	False
--use_wandb	Enable WandB logging	False

 Outputs
Saved models: models/encoder.pth, models/decoder.pth

Training logs printed and optionally sent to Weights & Biases

 Example
python-repl
Copy
Edit
Epoch 1/10, Loss: 3.1240
Epoch 2/10, Loss: 2.7152
...
 Model saved in `models/`
 Contributing
Pull requests and suggestions are welcome! For major changes, please open an issue first.



Let me know if you want a `requirements.txt` or example inference script to go along with this.

