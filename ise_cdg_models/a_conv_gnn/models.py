import random
import typing

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import vocab
from torch_geometric import nn as geo_nn

from ise_cdg_models.reusable_modules import BeforeRNNAttention


class CodeSequenceEncoder(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = embedding
        self.gru = nn.GRU(
            embedding_size, hidden_size,
        )

        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
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

        outputs, hidden = self.gru(embedding)
        # outputs: (seq, batch, hidden [x2 if bidirectional])
        # hidden: (layers [x2 if bidirectional], batch, hidden)

        hidden = self.reshape_hidden_for_linear(hidden)
        # hidden: (1, batch, layers [x2 if bidirectional] x hidden)

        hidden = self.fc_hidden(hidden)

        # hidden: (1, batch, hidden)
        return outputs, hidden

class CodeASTEncoder(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = embedding
        self.gcn_conv = geo_nn.GCNConv(embedding_size, embedding_size)
        self.gru = nn.GRU(
            embedding_size, hidden_size,
        )

    def forward(self, v, e, batch_ind):
        # sum_v = sum(graph.v for graph in batch)
        # v: (sum_v), e: (2, sum_v)
        v_embed = self.dropout(self.embedding(v))
        v_embed = v_embed.squeeze(1) # : (sum_v, embed)
        context = self.gcn_conv(v_embed, e) # : (sum_v, embed)
        context_mean_pool = geo_nn.pool.global_mean_pool(context, batch_ind) # : (batch, embed)
        context_max_pool = geo_nn.pool.global_max_pool(context, batch_ind) # : (batch, embed)
        context_mean_pool, context_max_pool = context_mean_pool.unsqueeze(0), context_max_pool.unsqueeze(0)
        context_merged = torch.cat((context_mean_pool, context_max_pool), dim=0) # : (2, batch, embed)
        output, hidden = self.gru(context_merged)
        # output: (2, batch, hidden)
        # hidden: (1, batch, hidden)

        return output, hidden

class DocumentDecoder(nn.Module):
    def __init__(
            self, vocab_size, embedding_size, hidden_size, p,
            code_seq_attention, code_ast_attention, use_glove=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p)
        if use_glove:
            glove = vocab.GloVe(name='6B', dim=embedding_size)
            glove_weights = torch.FloatTensor(glove.vectors)
            self.embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(
            hidden_size * 2 + embedding_size, hidden_size * 2,
        )
        self.code_seq_attention, self.code_ast_attention = code_seq_attention, code_ast_attention

        self.fc1 = nn.Linear(hidden_size * 2, vocab_size)


    def forward(self, x, code_seq_states, code_ast_states, hidden):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))  # : (1, batch, embed)

        code_seq_context_vector = self.code_seq_attention(hidden, code_seq_states) # : (1, batch, hidden)
        code_ast_context_vector = self.code_ast_attention(hidden, code_ast_states) # : (1, batch, hidden)
        rnn_inp = torch.cat((code_seq_context_vector, code_ast_context_vector, embedding), dim=2)
        # rnn_inp: (1, batch, hidden + hidden + embed)

        outputs, hidden = self.gru(rnn_inp, hidden)
        # outputs: (1, batch, hidden), hidden: (1, batch, hidden * 2)
        predictions = self.fc1(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden

class AConvGNN(nn.Module):
    def __init__(
            self,
            src_vocab_size, src_embedding_size,
            md_vocab_size, md_embedding_size,
            hidden_size, use_glove=True,
    ):
        super().__init__()
        src_embedding = nn.Embedding(src_vocab_size, src_embedding_size)
        self.src_vocab_size, self.md_vocab_size = src_vocab_size, md_vocab_size
        self.code_seq_encoder = CodeSequenceEncoder(
            src_embedding, src_embedding_size, hidden_size, .5
        )
        self.code_ast_encoder = CodeASTEncoder(
            src_embedding, src_embedding_size, hidden_size, .5
        )
        self.document_decoder = DocumentDecoder(
            md_vocab_size, md_embedding_size, hidden_size, .5,
            BeforeRNNAttention(hidden_size, 2 * hidden_size),
            BeforeRNNAttention(hidden_size, 2 * hidden_size),
            use_glove=use_glove,
        )

    def forward(
            self,
            source,
            source_ast_nodes, source_ast_edges, batch_index,
            target,
            device,
            teacher_force_ratio=0.9,
        ):
        batch_size = source.shape[1]
        target_sequence_len = target.shape[0]
        target_vocab_size = self.md_vocab_size

        outputs = torch.zeros(
            target_sequence_len, batch_size, target_vocab_size
        ).to(device)

        code_seq_states, code_seq_hidden = self.code_seq_encoder(source)
        code_ast_states, code_ast_hidden = self.code_ast_encoder(source_ast_nodes, source_ast_edges, batch_index)
        hidden = torch.cat((code_seq_hidden, code_ast_hidden), dim=2)

        x = target[0]

        for t in range(1, target_sequence_len):
            output, hidden = self.document_decoder(x, code_seq_states, code_ast_states, hidden)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def generate_one_markdown(
        self,
        source: torch.Tensor,
        source_ast_nodes: torch.Tensor, source_ast_edges: torch.Tensor, batch_index: torch.Tensor,
        start_ind: int, eos_ind: int,
        sequence_max_length: int,
        device: torch.device,
        ):
        batch_size = source.shape[1]
        assert batch_size == 1, "batch size must be 1"

        outputs = torch.zeros(
            sequence_max_length,
        ).to(device)

        code_seq_states, code_seq_hidden = self.code_seq_encoder(source)
        code_ast_states, code_ast_hidden = self.code_ast_encoder(source_ast_nodes, source_ast_edges, batch_index)
        hidden = torch.cat((code_seq_hidden, code_ast_hidden), dim=2)

        x = torch.ones(batch_size) * start_ind
        x = x.to(device).long()
        outputs[0] = x
        sequence_length = 1
        for t in range(1, sequence_max_length):
            output, hidden = self.document_decoder(x, code_seq_states, code_ast_states, hidden)
            x = output.argmax(1)
            outputs[t] = x
            sequence_length = t + 1
            if outputs[t] == eos_ind:
                break

        return outputs[:sequence_length]
