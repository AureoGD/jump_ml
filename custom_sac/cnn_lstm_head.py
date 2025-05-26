import torch
import torch.nn as nn


class FeatureHead(nn.Module):
    """
    CNN (optionally CNN + LSTM) feature extractor for one observation branch.
    """

    def __init__(self, channels, time_steps, use_lstm=False):
        super().__init__()

        self.use_lstm = use_lstm

        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, time_steps)
            cnn_out = self.cnn(dummy)
            cnn_flat_size = cnn_out.shape[1] * cnn_out.shape[2]

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=cnn_flat_size,
                hidden_size=128,
                batch_first=True,
            )
            self.out_dim = 128
        else:
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cnn_flat_size, 128),
                nn.ReLU(),
            )
            self.out_dim = 128

    def forward(self, x):
        x = self.cnn(x)

        if self.use_lstm:
            x = x.flatten(start_dim=1).unsqueeze(1)  # (B, 1, F)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
        else:
            x = self.linear(x)

        return x
