import os
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from custom_sac.networks import ActorNetwork, CriticNetwork, ValueNetwork
from custom_sac.replay_buffer import ReplayBuffer


class Agent:

    def __init__(
        self,
        env,
        alpha="auto",
        beta=3e-4,
        gamma=0.99,
        tau=0.005,
        max_size=1000000,
        batch_size=256,
        reward_scale=2,
        chkpt_dir="tmp/sac",
        log_dir="runs/sac_logs",
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.memory = None
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.chkpt_dir = chkpt_dir

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        self.learn_step = 0

        # Observation and action space
        self.input_dims = self._process_observation_space(env.observation_space)
        self.is_dict_obs = isinstance(self.input_dims, dict)

        self.n_actions = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        # Networks
        self.actor = ActorNetwork(
            alpha if isinstance(alpha, float) else 3e-4,
            self.input_dims,
            self.max_action,
            n_actions=self.n_actions,
            chkpt_dir=chkpt_dir,
        )
        self.critic_1 = CriticNetwork(beta, self.input_dims, self.n_actions, chkpt_dir=chkpt_dir, name="critic_1")
        self.critic_2 = CriticNetwork(beta, self.input_dims, self.n_actions, chkpt_dir=chkpt_dir, name="critic_2")
        self.value = ValueNetwork(beta, self.input_dims, chkpt_dir=chkpt_dir, name="value")
        self.target_value = ValueNetwork(beta, self.input_dims, chkpt_dir=chkpt_dir, name="target_value")

        self.target_value.load_state_dict(self.value.state_dict())

        # Replay buffer
        self.memory = ReplayBuffer(
            max_size=max_size,
            obs_space_dict=self.input_dims if self.is_dict_obs else {"obs": self.input_dims},
            n_actions=self.n_actions,
        )

        # Entropy tuning
        if isinstance(alpha, str) and alpha == "auto":
            self.target_entropy = -np.prod(self.n_actions).item()
            self.log_alpha = T.tensor(np.log(0.1), requires_grad=True, device=self.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=beta)
            self.entropy_tuning = True
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.entropy_tuning = False
            self.alpha = T.tensor(float(alpha)).to(self.device)

        # Episode loss accumulators
        self.reset_episode_loss_trackers()

    def reset_episode_loss_trackers(self):
        self.episode_value_loss = 0.0
        self.episode_critic1_loss = 0.0
        self.episode_critic2_loss = 0.0
        self.episode_actor_loss = 0.0
        self.episode_entropy_loss = 0.0
        self.episode_learn_steps = 0

    def _process_observation_space(self, obs_space):
        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            return obs_space.shape
        elif hasattr(obs_space, "spaces"):
            return {k: v.shape for k, v in obs_space.spaces.items()}
        else:
            raise ValueError("Unsupported observation space type")

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()

        if self.is_dict_obs:
            obs = {k: T.tensor(v, dtype=T.float32).unsqueeze(0).to(self.actor.device) for k, v in observation.items()}
        else:
            obs = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        mu, sigma = self.actor.forward(obs)
        probabilities = Normal(mu, sigma)

        if evaluate:
            actions = mu
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.actor.device)
        self.actor.train()

        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        if self.is_dict_obs:
            state = {k: T.tensor(states[k], dtype=T.float32).to(self.actor.device) for k in states.keys()}
            state_ = {k: T.tensor(states_[k], dtype=T.float32).to(self.actor.device) for k in states_.keys()}
        else:
            state = T.tensor(states, dtype=T.float32).to(self.actor.device)
            state_ = T.tensor(states_, dtype=T.float32).to(self.actor.device)

        actions = T.tensor(actions, dtype=T.float32).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.actor.device)

        # Value Network Update
        value = self.value(state).view(-1)
        target_value = self.target_value(state_).view(-1)
        target_value = target_value * (1 - dones)

        with T.no_grad():
            mu_, sigma_ = self.actor(state_)
            probabilities = Normal(mu_, sigma_)
            actions_, log_probs = self._sample_action_and_log_prob(probabilities)

            q1_ = self.critic_1(state_, actions_)
            q2_ = self.critic_2(state_, actions_)
            critic_value_ = T.min(q1_, q2_).view(-1)

            value_target = critic_value_ - self.alpha * log_probs.view(-1)

        value_loss = F.mse_loss(value, value_target)

        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        # Actor Network Update
        mu, sigma = self.actor(state)
        probabilities = Normal(mu, sigma)
        actions_, log_probs = self._sample_action_and_log_prob(probabilities)

        q1_new = self.critic_1(state, actions_)
        q2_new = self.critic_2(state, actions_)
        critic_new = T.min(q1_new, q2_new).view(-1)

        actor_loss = (self.alpha * log_probs.view(-1) - critic_new).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Critic Networks Update
        q_hat = self.reward_scale * rewards + self.gamma * target_value

        q1_old = self.critic_1(state, actions).view(-1)
        q2_old = self.critic_2(state, actions).view(-1)

        critic_1_loss = F.mse_loss(q1_old, q_hat)
        critic_2_loss = F.mse_loss(q2_old, q_hat)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        total_critic_loss = critic_1_loss + critic_2_loss
        total_critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Entropy Loss
        if self.entropy_tuning:
            entropy_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().detach()
        else:
            entropy_loss = T.tensor(0.0)

        self.update_network_parameters()

        # Step-based logs
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.learn_step)
        self.writer.add_scalar("Loss/Critic1", critic_1_loss.item(), self.learn_step)
        self.writer.add_scalar("Loss/Critic2", critic_2_loss.item(), self.learn_step)
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.learn_step)
        self.writer.add_scalar("Loss/Entropy", entropy_loss.item(), self.learn_step)
        self.writer.add_scalar("Alpha", self.alpha.item(), self.learn_step)

        self.learn_step += 1

        # Accumulate for episode logs
        self.episode_value_loss += value_loss.item()
        self.episode_critic1_loss += critic_1_loss.item()
        self.episode_critic2_loss += critic_2_loss.item()
        self.episode_actor_loss += actor_loss.item()
        self.episode_entropy_loss += entropy_loss.item()
        self.episode_learn_steps += 1

    def log_episode_losses(self, episode):
        if self.episode_learn_steps == 0:
            return

        self.writer.add_scalar("LossPerEpisode/Value", self.episode_value_loss / self.episode_learn_steps, episode)
        self.writer.add_scalar("LossPerEpisode/Critic1", self.episode_critic1_loss / self.episode_learn_steps, episode)
        self.writer.add_scalar("LossPerEpisode/Critic2", self.episode_critic2_loss / self.episode_learn_steps, episode)
        self.writer.add_scalar("LossPerEpisode/Actor", self.episode_actor_loss / self.episode_learn_steps, episode)
        self.writer.add_scalar("LossPerEpisode/Entropy", self.episode_entropy_loss / self.episode_learn_steps, episode)

        self.reset_episode_loss_trackers()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        value_state_dict = dict(value_params)
        target_value_state_dict = dict(target_value_params)

        for name in value_state_dict:
            value_state_dict[name] = (tau * value_state_dict[name] + (1 - tau) * target_value_state_dict[name])

        self.target_value.load_state_dict(value_state_dict)

    def _sample_action_and_log_prob(self, dist):
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob -= T.log(1 - T.tanh(action).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        action = T.tanh(action) * T.tensor(self.max_action).to(self.actor.device)
        return action, log_prob
