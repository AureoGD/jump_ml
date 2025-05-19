import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNN3HFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)

        # Get input shapes
        past_channels, past_steps = observation_space["pred_past"].shape
        fut_channels, fut_steps = observation_space["pred_fut"].shape
        nonpred_channels, nonpred_steps = observation_space["nonpredictable"].shape

        # CNN + LSTM for past
        self.cnn_past = nn.Sequential(
            nn.Conv1d(past_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, past_channels, past_steps))
            cnn_out = self.cnn_past(dummy)
            past_feat_dim = cnn_out.shape[1] * cnn_out.shape[2]
        self.lstm_past = nn.LSTM(
            input_size=past_feat_dim, hidden_size=64, batch_first=True
        )

        # CNN + LSTM for future
        self.cnn_future = nn.Sequential(
            nn.Conv1d(fut_channels, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, fut_channels, fut_steps))
            cnn_out = self.cnn_future(dummy)
            fut_feat_dim = cnn_out.shape[1] * cnn_out.shape[2]
        self.lstm_future = nn.LSTM(
            input_size=fut_feat_dim, hidden_size=64, batch_first=True
        )

        # CNN + LSTM for nonpredictable
        self.cnn_nonpred = nn.Sequential(
            nn.Conv1d(nonpred_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, nonpred_channels, nonpred_steps))
            cnn_out = self.cnn_nonpred(dummy)
            nonpred_feat_dim = cnn_out.shape[1] * cnn_out.shape[2]
        self.lstm_nonpred = nn.LSTM(
            input_size=nonpred_feat_dim, hidden_size=64, batch_first=True
        )

        # Final fusion layer
        self.fusion = nn.Sequential(nn.Linear(64 * 3, 256), nn.ReLU())

        self._features_dim = 256

    def forward(self, obs):
        def encode_branch(cnn, lstm, x):
            x = cnn(x).flatten(start_dim=1).unsqueeze(1)  # (B, 1, F)
            x_out, _ = lstm(x)
            return x_out[:, -1, :]  # (B, H)

        past_enc = encode_branch(self.cnn_past, self.lstm_past, obs["pred_past"])
        fut_enc = encode_branch(self.cnn_future, self.lstm_future, obs["pred_fut"])
        nonpred_enc = encode_branch(
            self.cnn_nonpred, self.lstm_nonpred, obs["nonpredictable"]
        )

        combined = torch.cat([past_enc, fut_enc, nonpred_enc], dim=1)
        return self.fusion(combined)
