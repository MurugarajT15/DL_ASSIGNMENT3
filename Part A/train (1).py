import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import os
import wandb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Character vocabulary and utility functions
class Lang:
    def __init__(self):
        self.char2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.index2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
        self.n_chars = 3

    def add_word(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1

    def word2indices(self, word):
        return [self.char2index[char] for char in word] + [self.char2index['<EOS>']]

    def indices2word(self, indices):
        return ''.join([self.index2char[idx] for idx in indices if idx not in [0, 1, 2]])

# Dataset class
class TransliterationDataset(Dataset):
    def __init__(self, path, input_lang, output_lang):
        self.pairs = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                eng, tam = line.strip().split('\t')
                input_lang.add_word(eng)
                output_lang.add_word(tam)
                self.pairs.append((eng, tam))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

# Collate function
def collate_fn(batch, input_lang, output_lang):
    input_seqs = [torch.tensor(input_lang.word2indices(pair[0]), dtype=torch.long) for pair in batch]
    target_seqs = [torch.tensor(output_lang.word2indices(pair[1]), dtype=torch.long) for pair in batch]
    
    input_lengths = [len(seq) for seq in input_seqs]
    target_lengths = [len(seq) for seq in target_seqs]
    
    input_padded = nn.utils.rnn.pad_sequence(input_seqs, padding_value=0)
    target_padded = nn.utils.rnn.pad_sequence(target_seqs, padding_value=0)
    
    return input_padded, target_padded, input_lengths, target_lengths

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.1, cell_type='GRU', bidirectional=False):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        rnn_cls = getattr(nn, cell_type)
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, hidden

# Attention (dot-product)
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(2)
        attn_scores = torch.bmm(encoder_outputs.permute(1, 0, 2), hidden).squeeze(2)
        attn_weights = self.softmax(attn_scores)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2)).squeeze(1)
        return context

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers=1, dropout=0.1, cell_type='GRU', use_attention=False):
        super(Decoder, self).__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = getattr(nn, cell_type)(embed_size + (hidden_size if use_attention else 0), hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size) if use_attention else None

    def forward(self, input, hidden, encoder_outputs=None):
        embedded = self.embedding(input).unsqueeze(0)
        if self.use_attention:
            context = self.attention(hidden, encoder_outputs)
            embedded = torch.cat((embedded, context.unsqueeze(0)), dim=2)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.out(output.squeeze(0))
        return prediction, hidden

# Seq2Seq Wrapper
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len = trg.shape[0]
        batch_size = trg.shape[1]
        vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src, [len(x) for x in src])
        input = trg[0, :]  # <SOS>

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs if self.decoder.use_attention else None)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# Training function
def train_model(args):
    input_lang = Lang()
    output_lang = Lang()
    dataset = TransliterationDataset(args.data_path, input_lang, output_lang)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, input_lang, output_lang))

    encoder = Encoder(input_lang.n_chars, args.embedding_size, args.hidden_size, args.n_layers, args.dropout, args.cell_type, args.bidirectional).to(device)
    decoder = Decoder(output_lang.n_chars, args.embedding_size, args.hidden_size * (2 if args.bidirectional else 1),
                      args.n_layers, args.dropout, args.cell_type, args.attention).to(device)
    
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if args.use_wandb:
        wandb.init(project="transliteration", config=vars(args))
        wandb.watch(model)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for src, trg, src_lengths, trg_lengths in dataloader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=0.5)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        if args.use_wandb:
            wandb.log({'loss': epoch_loss})

    os.makedirs('models', exist_ok=True)
    torch.save(encoder.state_dict(), 'models/encoder.pth')
    torch.save(decoder.state_dict(), 'models/decoder.pth')
    print("✅ Model saved in `models/`")

# Argparse entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='eng_ita.txt')
    parser.add_argument('--cell_type', type=str, choices=['RNN', 'LSTM', 'GRU'], default='GRU')
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()
    train_model(args)
    print("✅ Training complete.")
    if args.use_wandb:
        wandb.finish()