from turtle import forward
import typing
from torch import nn


class FeaturesEncoder(nn.Module):
    def __init__(self, features_length: int, context_size: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(features_length, features_length)
        self.fc2 = nn.Linear(features_length, context_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        # features.shape: (batch, features_length)
        features = self.dropout(features)
        features = self.relu(self.fc1(features))
        features = self.relu(self.fc2(features))
        return features