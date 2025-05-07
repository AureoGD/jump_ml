# rl_training/policies/mlp_policy.py

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy


class MLPPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        policy_kwargs = kwargs.pop("policy_kwargs", {})

        net_arch = policy_kwargs.get("net_arch", [256, 256])

        # Default activation function
        activation_fn = nn.ReLU  # always use ReLU unless you want something else

        kwargs.update(dict(net_arch=net_arch, activation_fn=activation_fn))

        super().__init__(*args, **kwargs)


class MLPSACPolicy(SACPolicy):
    """
    SAC Policy using a customizable MLP (no CNN).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
