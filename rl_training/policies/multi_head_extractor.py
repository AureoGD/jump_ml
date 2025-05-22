import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .cnn_ltsm_head import FeatureHead


class MultiHeadFeatureExtractor(BaseFeaturesExtractor):
    """
    Multi-branch CNN+LSTM feature extractor.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)

        def shape(key):
            return observation_space[key].shape  # (channels, time_steps)

        self.heads = nn.ModuleDict({
            "base_past": FeatureHead(*shape("base_past")),
            "base_future": FeatureHead(*shape("base_future")),
            "joint_past": FeatureHead(*shape("joint_past")),
            "joint_future": FeatureHead(*shape("joint_future")),
            "comp_past": FeatureHead(*shape("comp_past")),
        })

        total_dim = sum(h.out_dim for h in self.heads.values())

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
        )

        self._features_dim = 256

    def forward(self, obs):
        encodings = [head(obs[key]) for key, head in self.heads.items()]
        fused = torch.cat(encodings, dim=1)
        return self.fusion(fused)
