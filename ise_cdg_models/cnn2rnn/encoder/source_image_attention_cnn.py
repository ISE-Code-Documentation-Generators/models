from torch import nn


class SourceImageAttentionCNN(nn.Module):
    output_hidden_size = 16*16
    context_hidden_size = 512
    
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(21, 21),
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(31, 31),
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(29, 29),
        )

        # see the forward function to understand the shapes
        self.fc1 = nn.Linear(64 * 16 * 16, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)

        """
        - makes the frame size half.
        - pool has no learning parameter and can be reused.
        """
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2),
        )
        self.output_flatten = nn.Flatten(start_dim=2)
        self.context_flatten = nn.Flatten()
        self.dropout = nn.Dropout(.75)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x.shape: [batch, seq_len, embedding_size]
        assert x.shape[1] == x.shape[2] == 320
        x = x.unsqueeze(1)  # shape: [batch, 1, seq_len, embedding_size]
        for conv_id in range(1, 4):
            conv_kernel = getattr(self, f'conv{conv_id}')
            x = self.pool(self.tanh(conv_kernel(x)))

        context = self.context_flatten(x) # : (batch, kernel_out x seq_len_remain x embedding_size_remain)
        for fc_id in range(1, 5):
            fc = getattr(self, f'fc{fc_id}')
            context = self.relu(fc(context))

        return self.dropout(x), self.dropout(context)  # : (batch, kernel_out, seq_len_remain x embedding_size_remain) , (batch, encoder_context_size),