from abc import abstractmethod
from enum import Enum
from typing import Any, Union
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torchtext import vocab


class VocabEmbeddingHelper:
    class VectorsType(Enum):
        GLOVE_6B = "GloVe 6B"
        SIMPLE = "simple"

    def __init__(self, vocab: vocab.Vocab) -> None:
        self.adaptee_vocab = vocab

    @property
    def vocab_size(self):
        return len(self.adaptee_vocab)

    def get_embedding(
        self, embedding_size, vectors_type: "VocabEmbeddingHelper.VectorsType"
    ) -> "nn.Embedding":
        if vectors_type == self.VectorsType.GLOVE_6B:
            glove_vocab = vocab.GloVe(name="6B", dim=embedding_size)
            tokens_list = [
                self.adaptee_vocab.get_itos()[i] for i in range(self.vocab_size)
            ]
            glove_weights_subset= glove_vocab.get_vecs_by_tokens(tokens_list)
            return nn.Embedding.from_pretrained(glove_weights_subset, freeze=False)
        elif vectors_type == self.VectorsType.SIMPLE:
            return nn.Embedding(self.vocab_size, embedding_size)