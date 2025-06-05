# switch_rules/nn_switch_rule.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming SwitchRuleBase is in jump_ga.env_ga.switch_rule_base
# Adjust import path as needed
from .switch_rule_base import SwitchRuleBase


class SimpleMLP(SwitchRuleBase, nn.Module):

    def __init__(self, observation_dim: int, num_modes: int, hidden_dim: int = 64):
        """
        A simple MLP-based neural network to select an MPC mode.

        Args:
            observation_dim (int): Dimensionality of the input state vector.
            num_modes (int): Number of discrete MPC modes to choose from.
            hidden_dim (int): Size of the hidden MLP layers.
        """
        # Call both parent constructors
        SwitchRuleBase.__init__(self)  # You might pass gym.spaces here in the future
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_modes)  # Outputs logits for each mode

        # Store for convenience, especially if observation/action_space not passed to base
        self.observation_dim = observation_dim
        self.num_modes = num_modes

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            state_tensor (torch.Tensor): The input state.

        Returns:
            torch.Tensor: Logits for each mode.
        """
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        mode_logits = self.fc_out(x)
        return mode_logits

    def get_mode(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Selects an MPC mode based on the current state using the NN.

        Args:
            state (np.ndarray): The current robot state.
            deterministic (bool): If True, returns the mode with the highest probability (argmax).
                                  If False, samples from the probability distribution.

        Returns:
            int: The selected MPC mode.
        """
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        else:
            state_tensor = state.unsqueeze(0) if state.ndim == 1 else state

        # Ensure the tensor is on the same device as the model parameters
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():  # Important for inference
            mode_logits = self.forward(state_tensor)
            mode_probabilities = F.softmax(mode_logits, dim=-1)

        if deterministic:
            selected_mode = torch.argmax(mode_probabilities, dim=-1).item()
        else:
            # Sample from the distribution
            selected_mode = torch.multinomial(mode_probabilities, num_samples=1).item()

        return selected_mode

    def reset(self):
        # This simple MLP doesn't have internal states to reset,
        # but a more complex recurrent NN might.
        pass


# Example Usage (conceptual):
# observation_dim = 10 # Example: Replace with your actual state dimension
# num_rgo_modes = 3    # Example: Based on your RGC controller (OptProblem0, 1, 2)
#
# nn_switcher = SimpleNNSwitchRule(observation_dim, num_rgo_modes)
#
# # Get some robot state (as a numpy array)
# current_robot_state = np.random.rand(observation_dim)
#
# # Get a mode (stochastically)
# chosen_mode = nn_switcher.get_mode(current_robot_state)
# print(f"Stochastically Chosen Mode: {chosen_mode}")
#
# # Get a mode (deterministically)
# chosen_mode_det = nn_switcher.get_mode(current_robot_state, deterministic=True)
# print(f"Deterministically Chosen Mode: {chosen_mode_det}")
