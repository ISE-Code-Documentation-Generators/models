import torch
from torch import nn


class BeforeRNNAttention(nn.Module):

    def __init__(self, ehs, dhs) -> None:
        super().__init__()
        self.ehs, self.dhs = ehs, dhs
        self.energy = nn.Linear(self.ehs + self.dhs, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, si_1, h):
        # si_1: (1, b, dhs) # dhs: decoder hidden size
        # h: (esl, b, ehs)  # esl: encoder sequence length, ehs: encoder hidden size

        esl = h.shape[0]
        si_1_reshaped = si_1.repeat(esl, 1, 1)  # : (esl, b, dhs)
        energy_inp = torch.cat((si_1_reshaped, h), dim=2)  # : (seq, batch, dhs + ehs)
        energy = self.relu(self.energy(energy_inp))  # : (seq, batch, 1)
        attention = self.softmax(energy)  # : (seq, batch, 1)
        context_vector = torch.einsum("snk,snl->knl", attention, h)
        # context_vector: (1, batch, ehs)
        return context_vector
