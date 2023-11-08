from torch import nn


class SourceImageCNN(nn.Module):
    def __init__(self, context_size, conv_flatten_size):
      super().__init__()

      convolution_properties = {'padding':(0, 0), 'kernel_size':(17, 17), 'stride':(1, 1), 'dilation':(1, 1)}
      pooling_properties = {'kernel_size':(2, 2), 'stride':(2, 2), 'padding':(0, 0), 'dilation':(1, 1)}

      self.conv1 = nn.Conv2d(
          in_channels=1, out_channels=8, **convolution_properties,
      )
      self.conv2 = nn.Conv2d(
          in_channels=8, out_channels=16, **convolution_properties,
      )
      self.conv3 = nn.Conv2d(
          in_channels=16, out_channels=24, **convolution_properties,
      )
      self.conv4 = nn.Conv2d(
          in_channels=24, out_channels=32, **convolution_properties,
      )

      # see the forward function to understand the shapes
      # self.fc1 = nn.Linear(conv_flatten_size, 2048)
      self.fc2 = nn.Linear(conv_flatten_size, 512)
      self.fc3 = nn.Linear(512, context_size)

      """
        - makes the frame size half.
        - pool has no learning parameter and can be reused.
      """
      self.pool = nn.MaxPool2d(kernel_size=pooling_properties['kernel_size'], stride=pooling_properties['stride'])
      self.flatten = nn.Flatten()
      self.dropout = nn.Dropout(.5)
      self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape: [batch, seq_len, embedding_size]
        x = x.unsqueeze(1)  # shape: [batch, 1, seq_len, embedding_size]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)  # shape: (batch, flatten_size)
        # x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.dropout(x)  # shape: (batch, encoder_context_size)