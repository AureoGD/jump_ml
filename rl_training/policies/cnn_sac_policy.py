# rl_training/policies/cnn_sac_policy.py

import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy
from rl_training.policies.sac_feature_extractor import DictFeatureExtractor


class CNNSACPolicy(SACPolicy):
    """
    SAC Policy using CNN feature extractor.
    """

    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = kwargs.pop(
            "features_extractor_kwargs",
            {
                "conv_layers": [32, 64],
                "kernel_size": 3,
                "activation": "relu",
            },
        )

        super().__init__(
            *args,
            features_extractor_class=DictFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )
