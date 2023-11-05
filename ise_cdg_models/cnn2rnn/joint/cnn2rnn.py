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
    ):
        super().__init__()
        self.embedding = nn.Embedding(embed_size, vocab_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoder = SourceImageCNN(image_output_size, conv_flatten_size)
        self.image_output_size = image_output_size
        self.output_size = output_size
        self.rnn_processor = rnn_module(self.embed_size + self.image_output_size, self.hidden_size)
        self.ff_processor = ff_module(self.hidden_size, self.output_size)

    def forward(self, image, caption: str):
        # image.shape: (batch, *image_dimensions)
        # caption.shape: (batch, sequence_length)
        outputs = self.forward_procedure(image, caption)
        # outputs.shape: (batch, seq_length, output_size)

        if self.training:
            return outputs  # (batch, seq_length, output_size)
        else:
            return outputs[:, :1]  # (batch, 1, output_size)

    def caption(self, image, max_length: int, end_index: int):
        # image.shape: image_dimensions
        start_token_index = 1
        caption = torch.tensor([start_token_index]).to(
            device
        )  # shape: (sequence_length)

        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    image, caption.unsqueeze(0)
                )  # shape: (1, 1, output_size)
                word = outputs.argmax(-1)  # shape: (1, 1)
                caption = tensor_append(caption, word.item())

                if word.item() == end_index:
                    break

        return caption.tolist()

    def __image_to_par_input_image__(self, image, sequence_length=1):
        # image.shape: (batch, image_output_size)
        # shape: (batch, sequence_length, image_output_size)
        processed_image = image.unsqueeze(1).repeat_interleave(sequence_length, dim=1)
        return processed_image

    @staticmethod
    def __get_sequence_length__(caption):
        # caption.shape: (batch_size, sequence_length, *)
        return caption.shape[1]

    def forward_procedure(self, image, caption):
        # shape: (batch, sequence_length, embed_size)
        embedding = self.embedder(caption)

        # shape: (batch, image_output_size)
        processed_image = self.image_processor(image)
        # shape: (batch, sequence_length, image_output_size)
        processed_image = self.__image_to_par_input_image__(
            processed_image, self.__get_sequence_length__(caption)
        )

        # shape: (batch, sequence_length, embed_size + image_output_size)
        rnn_input = torch.cat((embedding, processed_image), dim=-1)
        # shape: (batch, sequence_length, hidden_size)
        outputs, _ = self.rnn_processor(rnn_input)
        # shape: (batch, sequence_length, ff_output_size)
        return self.ff_processor(outputs)
