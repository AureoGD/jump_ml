import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


# Placeholder for FeatureHead if cnn_lstm_head.py is not defined by the user.
# The user should replace this with their actual FeatureHead implementation if complex encoders are needed.
class FeatureHead(nn.Module):

    def __init__(self, channels, time_steps, use_lstm=False):
        super().__init__()
        # This is a very basic example.
        # User needs to ensure this matches their observation part's shape and desired processing.
        # Example: for an observation part of shape (C, H, W) or (C, L)
        # This example assumes input like (batch, channels, sequence_length/feature_dim)
        # For a simple vector feature (e.g. shape (F,)), channels=1, time_steps=F

        if channels == 0 or time_steps == 0:
            self.out_dim = 0
            self.fc_out_layer = nn.Identity()
            return

        # This is a placeholder. For image data (e.g., C, H, W), you'd use Conv2d.
        # For time-series/vector data (e.g., C, L where C is feature_dim, L is sequence_len,
        # or C=1, L=feature_dim for a flat vector), Conv1d might be applicable.
        current_out_dim = 0
        if channels == 1:  # Assuming flat vector of length time_steps
            self.base_layer = nn.Linear(time_steps, 128)
            current_out_dim = 128
        else:  # Fallback for multi-channel, could be Conv1d, etc.
            self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, padding=1)
            current_out_dim = 32 * time_steps
            self.flatten = nn.Flatten()

        if use_lstm:
            self.lstm = nn.LSTM(current_out_dim, 64, batch_first=True)
            self.out_dim = 64
        else:
            self.fc_out_layer = nn.Linear(current_out_dim, 64)
            self.out_dim = 64

    def forward(self, x):
        if hasattr(self, 'conv1'):
            x = F.relu(self.conv1(x))
            x = self.flatten(x)
        elif hasattr(self, 'base_layer'):
            # If input x is (Batch, Channels=1, Features), remove the channel dim for Linear layer
            if x.ndim == 3 and x.shape[1] == 1:
                x = x.squeeze(1)
            x = F.relu(self.base_layer(x))
        elif hasattr(self, 'fc_out_layer') and self.out_dim == 0:  # Case where channels/timesteps were 0
            return self.fc_out_layer(x)

        if hasattr(self, 'lstm'):
            if x.ndim == 2:
                x = x.unsqueeze(1)
            x, _ = self.lstm(x)
            x = x.squeeze(1)
        elif hasattr(self, 'fc_out_layer'):
            x = F.relu(self.fc_out_layer(x))
        return x


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureEncoder(nn.Module):

    def __init__(self, obs_shapes, use_encoder=True):
        super().__init__()
        self.use_encoder = use_encoder

        if self.use_encoder:
            if not isinstance(obs_shapes, dict):
                raise ValueError("If use_encoder is True, obs_shapes must be a dictionary of {key: shape_tuple}")

            self.heads = nn.ModuleDict()
            total_out_dim = 0
            for key, shape in obs_shapes.items():
                if not shape:  # Handle empty shape tuple if it occurs
                    print(f"Warning: Empty shape for key '{key}' in obs_shapes. Skipping head creation.")
                    continue
                channels = shape[0] if len(shape) > 1 else 1
                time_steps = shape[1] if len(shape) > 1 else shape[0]
                if len(shape) == 3:
                    channels = shape[0]
                    time_steps = np.prod(shape[1:])
                elif len(shape) == 0:  # Should not happen if previous check is there
                    print(f"Warning: Shape for key '{key}' is empty {shape}. Setting dummy dims for head.")
                    channels, time_steps = 0, 0

                head = FeatureHead(channels=channels, time_steps=time_steps, use_lstm=False)
                self.heads[key] = head
                total_out_dim += head.out_dim
            self.output_dim = total_out_dim
        else:
            if not isinstance(obs_shapes, tuple) or len(obs_shapes) == 0:
                raise ValueError(
                    "For flat obs (use_encoder=False), obs_shapes should be a tuple like (obs_dim,), got: {}".format(
                        obs_shapes))
            self.output_dim = obs_shapes[0]

    def forward(self, obs):
        if self.use_encoder:
            encoded_parts = []
            for key in self.heads.keys():
                head_input = obs[key]
                # Example logic for when a flat vector (F,) was passed as shape for head (channels=1, time_steps=F)
                # The head might expect input (B, 1, F)
                if head_input.ndim == 2 and hasattr(self.heads[key],
                                                    'base_layer'):  # If it's a flat vector for an FC based head
                    pass  # FC head can take (B,F)
                elif head_input.ndim == 2 and (hasattr(self.heads[key], 'conv1')
                                               and self.heads[key].conv1.in_channels == 1):
                    head_input = head_input.unsqueeze(1)

                encoded_parts.append(self.heads[key](head_input))
            if not encoded_parts:
                return T.tensor([], device=obs[list(obs.keys())[0]].device if obs else 'cpu')  # Handle case of no heads
            return T.cat(encoded_parts, dim=1)
        else:
            return obs


class CriticNetwork(nn.Module):

    def __init__(
        self,
        beta,
        obs_shapes,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="critic",
        chkpt_dir="tmp/sac",
        use_encoder=True,
    ):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file_base = os.path.join(self.chkpt_dir, self.name + "_sac")

        self.encoder = FeatureEncoder(obs_shapes, use_encoder=use_encoder)
        input_dim = self.encoder.output_dim + n_actions

        self.fc1 = init_layer(nn.Linear(input_dim, fc1_dims))
        self.fc2 = init_layer(nn.Linear(fc1_dims, fc2_dims))
        self.q = init_layer(nn.Linear(fc2_dims, 1), std=1.0)

        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = self.encoder(state)
        x = T.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.q(x)
        return q_val

    def save_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        state_dict = T.load(filepath, map_location=self.device)
        self.load_state_dict(state_dict)


class ActorNetwork(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(
        self,
        alpha,
        obs_shapes,
        max_action,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="actor",
        chkpt_dir="tmp/sac",
        use_encoder=True,
    ):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file_base = os.path.join(self.chkpt_dir, self.name + "_sac")

        self.encoder = FeatureEncoder(obs_shapes, use_encoder=use_encoder)
        input_dim = self.encoder.output_dim

        self.fc1 = init_layer(nn.Linear(input_dim, fc1_dims))
        self.fc2 = init_layer(nn.Linear(fc1_dims, fc2_dims))

        self.mu_layer = init_layer(nn.Linear(fc2_dims, n_actions), std=0.01)
        self.log_std_layer = init_layer(nn.Linear(fc2_dims, n_actions), std=0.01)

        self.max_action_val = float(max_action)
        self.reparam_noise = 1e-6

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = T.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        sigma = T.exp(log_std)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        actions_gaussian = probabilities.rsample() if reparameterize else probabilities.sample()
        actions_tanh = T.tanh(actions_gaussian)
        log_probs_gaussian = probabilities.log_prob(actions_gaussian)
        log_probs_tanh = log_probs_gaussian - T.log(1.0 - actions_tanh.pow(2) + self.reparam_noise)
        log_probs_tanh = log_probs_tanh.sum(dim=1, keepdim=True)
        scaled_tanh_action = actions_tanh * self.max_action_val
        return scaled_tanh_action, log_probs_tanh

    def save_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        state_dict = T.load(filepath, map_location=self.device)
        self.load_state_dict(state_dict)
