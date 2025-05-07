# rl_training/policies/sac_feature_extractor.py

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DictFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations:
    - Applies CNN on 'predictable' features
    - Applies flattening on 'nonpredictable' features
    - Concatenates both
    """

    def __init__(
        self, observation_space, conv_layers=[32, 64], kernel_size=3, activation="relu"
    ):
        # Call parent class
        super().__init__(observation_space, features_dim=1)  # placeholder, fix later

        activation_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # Access Dict spaces
        predictable_space = observation_space.spaces["predictable"]
        nonpredictable_space = observation_space.spaces["nonpredictable"]

        n_input_channels = predictable_space.shape[0]  # features dimension
        seq_len = predictable_space.shape[1]  # time steps

        # CNN for predictable
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, conv_layers[0], kernel_size),
            activation_fn(),
            nn.Conv1d(conv_layers[0], conv_layers[1], kernel_size),
            activation_fn(),
        )

        # Flatten nonpredictable part
        nonpredictable_dim = 1
        for s in nonpredictable_space.shape:
            nonpredictable_dim *= s
        self.nonpredictable_dim = nonpredictable_dim

        # Calculate CNN output dimension dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, seq_len)
            cnn_output = self.cnn(sample_input)
            cnn_output_dim = cnn_output.view(1, -1).shape[1]

        # Total features dim
        self._features_dim = cnn_output_dim + nonpredictable_dim

    def forward(self, observations):
        predictable = observations["predictable"]
        nonpredictable = observations["nonpredictable"]

        # CNN branch
        cnn_features = self.cnn(predictable)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Flatten nonpredictable branch
        nonpredictable_features = nonpredictable.view(nonpredictable.size(0), -1)

        # Concatenate
        combined_features = torch.cat([cnn_features, nonpredictable_features], dim=1)
        return combined_features
