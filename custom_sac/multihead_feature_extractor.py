import torch
import torch.nn as nn
from gymnasium import spaces

from .cnn_lstm_head import FeatureHead


class MultiHeadFeatureExtractor(nn.Module):
    """
    Multi-branch CNN+LSTM feature extractor for Dict observations.
    Each key in the observation dict gets its own CNN (optionally with LSTM).
    """

    def __init__(self, observation_space: spaces.Dict, use_lstm=False):
        super().__init__()

        self.heads = nn.ModuleDict()

        def get_shape(key):
            return observation_space.spaces[key].shape  # (channels, time_steps)

        self.heads["base_past"] = FeatureHead(*get_shape("base_past"), use_lstm)
        self.heads["base_future"] = FeatureHead(*get_shape("base_future"), use_lstm)
        self.heads["joint_past"] = FeatureHead(*get_shape("joint_past"), use_lstm)
        self.heads["joint_future"] = FeatureHead(*get_shape("joint_future"), use_lstm)
        self.heads["comp_past"] = FeatureHead(*get_shape("comp_past"), use_lstm)

        # Calculate total output dimension
        self.latent_dim = sum(head.out_dim for head in self.heads.values())

        # Fusion layer (optional, could be linear or identity)
        self.fusion = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
        )
        self.output_dim = 256  # For clarity

    def forward(self, obs):
        encoded = [head(obs[key]) for key, head in self.heads.items()]
        fused = torch.cat(encoded, dim=1)
        return self.fusion(fused)
