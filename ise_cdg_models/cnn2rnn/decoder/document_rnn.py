import torch
from torch import nn
from torchtext import vocab

class DocumentDecoder(nn.Module):

    def __init__(
            self, 
            embedding, # could be from glove
            vocab_size, embed_size, hidden_size,
            encoder_context_size,
            ):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = embedding
        self.lstm = nn.LSTM(
            embed_size + encoder_context_size, hidden_size,
        )
        self.fc1 = nn.Linear(hidden_size, vocab_size)
    

    def forward(self, x, hidden, encoder_context):
        x = x.unsqueeze(0) 
        encoder_context = encoder_context.unsqueeze(1)
        # x, hidden, encoder_context: (1, batch), (1, batch, hidden), (1, batch, encoder_context_size)
        embedding = self.dropout(self.embedding(x))  # : (1, batch, embed)
        rnn_inp = torch.cat((embedding, encoder_context, embedding), dim=2)
        # rnn_inp: (1, batch, embed_size + encoder_context_size)

        outputs, hidden = self.lstm(rnn_inp, hidden)
        # outputs: (1, batch, hidden), hidden: (1, batch, hidden)
        predictions = self.fc1(outputs) # : (1, batch, vocab_size)
        predictions = predictions.squeeze(0) # : (batch, vocab_size)
        return predictions, hidden
