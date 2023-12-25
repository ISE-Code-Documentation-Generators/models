import random
import torch
from torch import nn

from ise_cdg_models.cnn2rnn.decoder import DocumentDecoder
from ise_cdg_models.cnn2rnn.embeddings import VocabEmbeddingHelper
from ise_cdg_models.cnn2rnn.encoder import SourceImageEncoder



class CNN2RNN(nn.Module):
    def __init__(
        self,
        src_vocab,
        md_vocab,
        hidden_size,
        encoder_context_size,
        conv_flatten_size,
        use_glove=True,
    ):
        super().__init__()
        src_embed_size = 512
        md_embed_size = 300
        src_vocab_helper = VocabEmbeddingHelper(src_vocab)
        md_vocab_helper = VocabEmbeddingHelper(md_vocab)
        self.md_vocab_size = md_vocab_helper.vocab_size
        self.encoder = SourceImageEncoder(
                src_vocab_helper.get_embedding(src_embed_size, src_vocab_helper.VectorsType.SIMPLE),
                encoder_context_size, conv_flatten_size)
        decoder_vectype = md_vocab_helper.VectorsType.GLOVE_6B if use_glove else md_vocab_helper.VectorsType.SIMPLE
        self.decoder = DocumentDecoder(
                md_vocab_helper.get_embedding(md_embed_size, decoder_vectype), 
                md_vocab_helper.vocab_size, md_embed_size, hidden_size, encoder_context_size,)

    def forward(self, source, markdown, device,
            teacher_force_ratio=0.9):
        batch_size = source.shape[1]
        target_sequence_len = markdown.shape[0]
        target_vocab_size = self.md_vocab_size
        
        outputs = torch.zeros(
            target_sequence_len, batch_size, target_vocab_size
        ).to(device)
        
        code_image_context = self.encoder(source) # shape: (batch, encoder_context_size)
        hidden = None

        x = markdown[0] # shape: (batch)
        
        for t in range(1, target_sequence_len):
            output, hidden = self.decoder(x, hidden, code_image_context)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = markdown[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    
    def generate_one_markdown(
        self,
        source: torch.Tensor,
        sos_ind: int, eos_ind: int,
        sequence_max_length: int,
        device: torch.device,
        ):
        batch_size = source.shape[1]
        assert batch_size == 1, "batch size must be 1"

        outputs = torch.zeros(
            sequence_max_length,
        ).to(device)

        code_image_context = self.encoder(source) # shape: (batch, encoder_context_size)
        hidden = None

        x = torch.ones(batch_size) * sos_ind
        x = x.to(device).long()
        outputs[0] = x
        sequence_length = 1
        for t in range(1, sequence_max_length):
            output, hidden = self.decoder(x, hidden, code_image_context)
            x = output.argmax(1)
            outputs[t] = x
            sequence_length = t + 1
            if outputs[t] == eos_ind:
                break

        return outputs[:sequence_length]
