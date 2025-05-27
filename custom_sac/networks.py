import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .cnn_lstm_head import FeatureHead


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureEncoder(nn.Module):

    def __init__(self, obs_shapes, use_encoder=True):
        super().__init__()

        self.use_encoder = use_encoder

        if self.use_encoder:
            self.heads = nn.ModuleDict({
                key: FeatureHead(channels=shape[0], time_steps=shape[1], use_lstm=False)
                for key, shape in obs_shapes.items()
            })
            self.output_dim = sum([head.out_dim for head in self.heads.values()])
        else:
            assert isinstance(obs_shapes, tuple), "For flat obs, obs_shapes should be a tuple (obs_dim,)"
            self.output_dim = obs_shapes[0]

    def forward(self, obs):
        if self.use_encoder:
            encoded = [self.heads[key](obs[key]) for key in self.heads.keys()]
            return T.cat(encoded, dim=1)
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

        self.encoder = FeatureEncoder(obs_shapes, use_encoder=use_encoder)
        input_dim = self.encoder.output_dim + n_actions

        self.fc1 = init_layer(nn.Linear(input_dim, fc1_dims))
        self.fc2 = init_layer(nn.Linear(fc1_dims, fc2_dims))
        self.q = init_layer(nn.Linear(fc2_dims, 1), std=1.0)

        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.checkpoint_file = os.path.join(chkpt_dir, name + "_sac")

    def forward(self, state, action):
        x = self.encoder(state)
        x = T.cat([x, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):

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

        self.encoder = FeatureEncoder(obs_shapes, use_encoder=use_encoder)
        input_dim = self.encoder.output_dim

        self.fc1 = init_layer(nn.Linear(input_dim, fc1_dims))
        self.fc2 = init_layer(nn.Linear(fc1_dims, fc2_dims))

        self.mu = init_layer(nn.Linear(fc2_dims, n_actions), std=0.01)  # Low init for stable output
        self.sigma = init_layer(nn.Linear(fc2_dims, n_actions), std=0.01)

        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.checkpoint_file = os.path.join(chkpt_dir, name + "_sac")

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1.0)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = T.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = dist.rsample()
        else:
            actions = dist.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)

        log_probs = dist.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
