from turtle import forward
import typing
from torch import nn
import torch


class FeaturesEncoder(nn.Module):
    def __init__(self, features_length: int, context_size: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(features_length, features_length)
        self.fc2 = nn.Linear(features_length, context_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        # features.shape: (features_length, batch)
        print("FeaturesEncoder")
        print(features.shape)
        features = torch.einsum('be->eb', features) 
        features = self.dropout(features)
        features = self.relu(self.fc1(features))
        print(features.shape)
        features = self.relu(self.fc2(features))
        return features