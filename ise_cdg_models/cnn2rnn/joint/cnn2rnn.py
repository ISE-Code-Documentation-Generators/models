import random
from ise_cdg_models.cnn2rnn.decoder.document_rnn import DocumentRNN
from ise_cdg_models.cnn2rnn.encoder.source_image_cnn import SourceImageCNN
import torch
from torch import nn




class ParImageCaptionerArchitecture(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        rnn_module,
        hidden_size,
        image_output_size,
        conv_flatten_size,
        ff_module,
        output_size,
        glove_weights=None,
    ):
        super().__init__()
        if glove_weights is None:
            embedding = nn.Embedding(embed_size, vocab_size)
        else:
            embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        self.encoder = SourceImageCNN(image_output_size, conv_flatten_size)
        self.decoder = DocumentRNN(embedding, vocab_size, embed_size, hidden_size, image_output_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.image_output_size = image_output_size
        self.output_size = output_size

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
