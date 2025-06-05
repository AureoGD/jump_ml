# switch_rules/nn_switch_rule.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming SwitchRuleBase is in jump_ga.env_ga.switch_rule_base
# Adjust import path as needed
from .switch_rule_base import SwitchRuleBase
from utils.state_normalizer import StateNormalizer


class SimpleMLP(SwitchRuleBase, nn.Module):

    def __init__(self, observation_dim: int, num_modes: int, state_normalizer=None, **kwargs):
        """
        An NN-based SwitchRule that handles its own state normalization.

        Args:
            observation_dim (int): The dimension of the (raw) observation space.
            num_modes (int): The number of discrete modes to output.
            state_normalizer (StateNormalizer, optional): An instance of a fitted StateNormalizer.
                                                        If None, states will be used as-is.
        """
        super().__init__()  # Calls both SwitchRuleBase and nn.Module __init__
        self.normalizer = state_normalizer
        self.num_modes = num_modes

        # The NN layers are defined based on the observation dimension
        self.fc1 = torch.nn.Linear(observation_dim, 128)  # Example hidden size
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc_out = torch.nn.Linear(128, num_modes)

    def forward(self, normalized_state_tensor: torch.Tensor) -> torch.Tensor:
        """The forward pass expects a pre-normalized tensor."""
        x = torch.relu(self.fc1(normalized_state_tensor))
        x = torch.relu(self.fc2(x))
        mode_logits = self.fc_out(x)
        return mode_logits

    def get_mode(self, raw_state: np.ndarray, deterministic: bool = False) -> int:
        """
        Takes a RAW state, normalizes it, and selects a mode.

        Args:
            raw_state (np.ndarray): The unnormalized state vector from the environment.
            deterministic (bool): If True, take the best action. If False, sample.

        Returns:
            int: The selected MPC mode.
        """
        # 1. Normalize the state if a normalizer is available
        if self.normalizer and self.normalizer.is_fitted:
            normalized_state = self.normalizer.transform(raw_state)
        else:
            normalized_state = raw_state  # Use raw state if no normalizer

        # 2. Convert to tensor
        state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0)
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        # 3. Get action from the NN
        with torch.no_grad():
            mode_logits = self.forward(state_tensor)
            mode_probabilities = torch.softmax(mode_logits, dim=-1)

        if deterministic:
            selected_mode = torch.argmax(mode_probabilities, dim=-1).item()
        else:
            selected_mode = torch.multinomial(mode_probabilities, num_samples=1).item()

        return selected_mode
