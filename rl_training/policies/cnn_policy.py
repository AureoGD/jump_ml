import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, conv_layers=[32, 64], kernel_size=3, activation="relu"
    ):
        super().__init__(observation_space, features_dim=1)

        activation_fn = nn.ReLU if activation == "relu" else nn.Tanh

        input_channels = observation_space["predictable"].shape[0]
        sequence_length = observation_space["predictable"].shape[1]

        layers = []
        last_channels = input_channels

        for filters in conv_layers:
            layers.append(nn.Conv1d(last_channels, filters, kernel_size))
            layers.append(activation_fn())
            last_channels = filters

        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, sequence_length)
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.view(1, -1).shape[1]

        self._features_dim = n_flatten

    def forward(self, observations):
        x = observations["predictable"]
        return self.cnn(x).flatten(start_dim=1)


class CNNPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {})

        kwargs.update(
            dict(
                features_extractor_class=CNNFeatureExtractor,
                features_extractor_kwargs=features_extractor_kwargs,
            )
        )

        super().__init__(*args, **kwargs)
