
import torch
from torch import nn
from ise_cdg_models.cnn2rnn.encoder.source_image_attention_cnn import SourceImageAttentionCNN, SourceImageAttentionCNNV2
from ise_cdg_models.cnn2rnn.encoder.source_image_cnn import SourceImageCNN

class SourceImageEncoder(nn.Module):
    def __init__(self, embedding, context_size, conv_flatten_size):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = embedding
        self.source_image_cnn = SourceImageCNN(context_size, conv_flatten_size)

    def forward(self, x: "torch.Tensor"):
        # x.shape: [seq_len, batch]
        print(x.shape)
        x = self.dropout(self.embedding(x)) # : [seq_len, batch, embedding_size]
        x = torch.einsum('sbe->bse', x) # : [batch, seq_len, embedding_size]
        context = self.source_image_cnn(x) # : (batch, encoder_context_size)
        return context
    

class SourceImageAttentionBasedEncoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = embedding
        self.source_image_cnn = SourceImageAttentionCNN()

    def forward(self, x):
        # x.shape: [seq_len, batch]
        x = self.dropout(self.embedding(x)) # : [seq_len, batch, embedding_size]
        x = torch.einsum('sbe->bse', x)
        output, context = self.source_image_cnn(x)
        output = torch.einsum('bsh->sbh', output) # : (fake_seq_len, batch, fake_hidden_size)
        context = context.unsqueeze(0) # : (1, b, conv_flatten_size)
        return output, context
    

class SourceImageAttentionBasedEncoderV2(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.dropout = nn.Dropout(.5)
        self.embedding = embedding
        self.source_image_cnn = SourceImageAttentionCNNV2()

    def forward(self, x):
        # x.shape: [seq_len, batch]
        x = self.dropout(self.embedding(x)) # : [seq_len, batch, embedding_size]
        x = torch.einsum('sbe->bse', x)
        output, context = self.source_image_cnn(x)
        output = torch.einsum('bsh->sbh', output) # : (fake_seq_len, batch, fake_hidden_size)
        context = context.unsqueeze(0) # : (1, b, conv_flatten_size)
        return output, context