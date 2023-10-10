import random
import typing

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import vocab

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            dropout=p,
        )

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.lstm(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_size, hidden_size, num_layers, p
    ):
        super().__init__()

        self.dropout = nn.Dropout(p)
        glove = vocab.GloVe(name='6B', dim=embedding_size)
        glove_weights = torch.FloatTensor(glove.vectors)
        self.embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            dropout=p,
            )
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        predictions = self.fc1(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size, src_embedding_size,
        md_vocab_size, md_embedding_size,
        hidden_size,
    ):
        super().__init__()
        self.src_vocab_size, self.md_vocab_size = src_vocab_size, md_vocab_size
        self.encoder = Encoder(
            src_vocab_size, src_embedding_size, hidden_size, 2, .4
            )
        self.decoder = Decoder(
            md_vocab_size, md_embedding_size, hidden_size, 2, .4
        )

    def forward(self, source, target, device, teacher_force_ratio=0.9):
        batch_size = source.shape[1]
        target_sequence_len = target.shape[0]
        target_vocab_size = self.md_vocab_size

        outputs = torch.zeros(
            target_sequence_len, batch_size, target_vocab_size
        ).to(device)

        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_sequence_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def generate_markdown(self, source, start_idx, device, sequence_length=50):
        batch_size = source.shape[1]

        outputs = torch.zeros(
            sequence_length, batch_size,
        ).to(device)

        hidden, cell = self.encoder(source)

        x = torch.ones(batch_size) * start_idx
        x = x.to(device).long()
        outputs[0] = x

        for t in range(1, sequence_length):
            output, hidden, cell = self.decoder(x, hidden, cell)
            x = output.argmax(1)
            outputs[t] = x

        return outputs

