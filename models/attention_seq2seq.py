import random
import typing

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import vocab


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size,
            bidirectional=True,
        )

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.flatten = nn.Flatten()

    def reshape_hidden_for_linear(self, h):
        # h: (layers [x2 if bidirectional], batch, hidden)
        h = torch.einsum('ijk->jik', h)
        # h: (batch, layers [x2 if bidirectional], hidden)
        h = self.flatten(h)
        # h: (batch, layers [x2 if bidirectional] x hidden)
        h = h.unsqueeze(0)
        # h: (1, batch, layers [x2 if bidirectional] x hidden)
        return h

    def forward(self, x):
        # x: (seq, batch)
        embedding = self.dropout(self.embedding(x))  # : (seq, batch, embed)

        outputs, (hidden, cell) = self.lstm(embedding)
        # outputs: (seq, batch, hidden [x2 if bidirectional])
        # hidden, cell: (layers [x2 if bidirectional], batch, hidden)

        hidden = self.reshape_hidden_for_linear(hidden)
        cell = self.reshape_hidden_for_linear(cell)
        # hidden, cell: (1, batch, layers [x2 if bidirectional] x hidden)

        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)

        # hidden, cell: (1, batch, hidden)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(
            self, vocab_size, embedding_size, hidden_size, p
    ):
        super().__init__()
        self.dropout = nn.Dropout(p)
        glove = vocab.GloVe(name='6B', dim=embedding_size)
        glove_weights = torch.FloatTensor(glove.vectors)
        self.embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            hidden_size * 2 + embedding_size, hidden_size,
        )

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))  # : (1, batch, embed)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)  # : (seq, batch, hidden)
        energy_inp = torch.cat((h_reshaped, encoder_states), dim=2)  # : (seq, batch, hidden + hidden [x2 if bidir])
        energy = self.relu(self.energy(energy_inp))  # : (seq, batch, 1)
        attention = self.softmax(energy)  # : (seq, batch, 1)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        # context_vector: (1, batch, hidden_size (x2 if bidir))

        rnn_inp = torch.cat((context_vector, embedding), dim=2)
        # rnn_inp: (1, batch, hidden_size (x2 if bidir) + embed)

        outputs, (hidden, cell) = self.lstm(rnn_inp, (hidden, cell))
        # outputs, hidden, cell: (1, batch, hidden)
        predictions = self.fc1(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class AttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size, src_embedding_size,
        md_vocab_size, md_embedding_size,
        hidden_size,
    ):
        super().__init__()
        self.src_vocab_size, self.md_vocab_size = src_vocab_size, md_vocab_size
        self.encoder = Encoder(
            src_vocab_size, src_embedding_size, hidden_size, .4
            )
        self.decoder = Decoder(
            md_vocab_size, md_embedding_size, hidden_size, .4
        )

    def forward(self, source, target, device, teacher_force_ratio=0.9):
        batch_size = source.shape[1]
        target_sequence_len = target.shape[0]
        target_vocab_size = self.md_vocab_size

        outputs = torch.zeros(
            target_sequence_len, batch_size, target_vocab_size
        ).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_sequence_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def generate_markdown(self, source, start_idx, device, sequence_length=50):
        batch_size = source.shape[1]

        outputs = torch.zeros(
            sequence_length, batch_size,
        ).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        x = torch.ones(batch_size) * start_idx
        x = x.to(device).long()
        outputs[0] = x

        for t in range(1, sequence_length):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            x = output.argmax(1)
            outputs[t] = x

        return outputs
