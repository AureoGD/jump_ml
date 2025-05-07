# stagnation_classifier/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNStagnationClassifier(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=4, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Use dummy input to calculate flatten size
        dummy_input = torch.zeros(1, in_channels, seq_len)
        x = self.conv1(dummy_input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 32, seq_len)
        x = F.relu(self.conv2(x))  # (batch, 64, seq_len)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
