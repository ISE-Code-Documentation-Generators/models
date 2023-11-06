
import torch
from torch import nn

from ise_cdg_models.cnn2rnn.encoder.source_image_cnn import SourceImageCNN

class SourceImageEncoder(nn.Module):
    def __init__(self, embedding, context_size, conv_flatten_size):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = embedding
        self.source_image_cnn = SourceImageCNN(context_size, conv_flatten_size)

    def forward(self, x):
        # x.shape: [batch, seq_len]
        x = self.dropout(self.embedding(x)) # : [batch, seq_len, embedding_size]
        context = self.source_image_cnn(x) # : (batch, encoder_context_size)
        return context