import torch
from torch import nn
from torchtext import vocab

from ise_cdg_models.reusable_modules import BeforeRNNAttention

class DocumentDecoderAttention(nn.Module):

    def __init__(
            self, 
            embedding, # could be from glove
            vocab_size, embed_size, hidden_size,
            encoder_hidden_size,
            ):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = embedding
        self.gru = nn.GRU(
            embed_size + encoder_hidden_size, hidden_size,
        )
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.attention = BeforeRNNAttention(ehs=encoder_hidden_size, dhs=hidden_size)
    

    def forward(self, x, hidden, encoder_output):
        x = x.unsqueeze(0) 
        # x, hidden, encoder_output: (1, batch), (1, batch, hidden), (esl, batch, ehs)
        embedding = self.dropout(self.embedding(x))  # : (1, batch, embed)
        attention_context = self.attention(hidden, encoder_output) # : (1, batch, ehs)
        rnn_inp = torch.cat((attention_context, embedding), dim=2)
        # rnn_inp: (1, batch, embed_size + ehs) s.t. ehs = encoder_hidden_size

        outputs, hidden = self.gru(rnn_inp, hidden)
        # outputs: (1, batch, hidden), hidden: (1, batch, hidden)
        predictions = self.fc1(outputs) # : (1, batch, vocab_size)
        predictions = predictions.squeeze(0) # : (batch, vocab_size)
        return predictions, hidden
