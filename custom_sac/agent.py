import os
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from custom_sac.networks import ActorNetwork, CriticNetwork
from custom_sac.replay_buffer import ReplayBuffer


class Agent:

    def __init__(
        self,
        env,
        obs_shapes,
        use_encoder=True,
        alpha="auto",
        beta=3e-4,
        gamma=0.99,
        tau=0.005,
        max_size=1_000_000,
        batch_size=256,
        reward_scale=1,
        chkpt_dir="tmp/sac",
        log_dir="runs/sac",
        gradient_steps=1,  # How many learning steps per environment step
        learning_starts=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.chkpt_dir = chkpt_dir
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.obs_shapes = obs_shapes
        self.use_encoder = use_encoder

        self.writer = SummaryWriter(log_dir=log_dir)
        self.learn_step = 0

        self.is_dict_obs = use_encoder
        self.n_actions = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        # Networks
        self.actor = ActorNetwork(
            alpha if isinstance(alpha, float) else 3e-4,
            obs_shapes,
            self.max_action,
            n_actions=self.n_actions,
            chkpt_dir=chkpt_dir,
            use_encoder=use_encoder,
        )
        self.critic_1 = CriticNetwork(beta,
                                      obs_shapes,
                                      self.n_actions,
                                      chkpt_dir=chkpt_dir,
                                      name="critic_1",
                                      use_encoder=use_encoder)
        self.critic_2 = CriticNetwork(beta,
                                      obs_shapes,
                                      self.n_actions,
                                      chkpt_dir=chkpt_dir,
                                      name="critic_2",
                                      use_encoder=use_encoder)
        self.target_critic_1 = CriticNetwork(beta,
                                             obs_shapes,
                                             self.n_actions,
                                             chkpt_dir=chkpt_dir,
                                             name="target_critic_1",
                                             use_encoder=use_encoder)
        self.target_critic_2 = CriticNetwork(beta,
                                             obs_shapes,
                                             self.n_actions,
                                             chkpt_dir=chkpt_dir,
                                             name="target_critic_2",
                                             use_encoder=use_encoder)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.memory = ReplayBuffer(
            max_size=max_size,
            obs_space_dict=obs_shapes if self.is_dict_obs else {"obs": obs_shapes},
            n_actions=self.n_actions,
        )

        # ✔️ Correct Entropy Tuning Initialization (Matching SB3)
        if isinstance(alpha, str) and alpha == "auto":
            self.target_entropy = -np.prod(env.action_space.shape).item()
            self.log_alpha = T.tensor(np.log(1.0), requires_grad=True, device=self.device)  # 🔥 Correct
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=beta)
            self.entropy_tuning = True
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.entropy_tuning = False
            self.alpha = T.tensor(float(alpha)).to(self.device)

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()

        if self.is_dict_obs:
            obs = {k: T.tensor(v, dtype=T.float32).unsqueeze(0).to(self.device) for k, v in observation.items()}
        else:
            obs = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.device)

        mu, sigma = self.actor.forward(obs)
        dist = Normal(mu, sigma)

        if evaluate:
            action = mu
        else:
            action = dist.rsample()

        action = T.tanh(action) * T.tensor(self.max_action).to(self.device)
        self.actor.train()

        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        if self.is_dict_obs:
            state_to_store = state
            new_state_to_store = new_state
        else:
            state_to_store = {"obs": state}
            new_state_to_store = {"obs": new_state}

        self.memory.store_transition(state_to_store, action, reward, new_state_to_store, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def learn(self):
        if self.memory.size() < max(self.learning_starts, self.batch_size):
            return

        for _ in range(self.gradient_steps):
            states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

            if self.is_dict_obs:
                state = {k: T.tensor(states[k], dtype=T.float32).to(self.device) for k in states.keys()}
                state_ = {k: T.tensor(states_[k], dtype=T.float32).to(self.device) for k in states_.keys()}
            else:
                state = T.tensor(states["obs"], dtype=T.float32).to(self.device)
                state_ = T.tensor(states_["obs"], dtype=T.float32).to(self.device)

            actions = T.tensor(actions, dtype=T.float32).to(self.device)
            rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
            dones = T.tensor(dones, dtype=T.float32).to(self.device)

            # ──────────────────────────────────────────
            # Critic Target Calculation
            # ──────────────────────────────────────────
            with T.no_grad():
                mu_, sigma_ = self.actor(state_)
                dist = Normal(mu_, sigma_)
                next_actions, log_probs = self._sample_action_and_log_prob(dist)

                target_q1 = self.target_critic_1(state_, next_actions).view(-1)
                target_q2 = self.target_critic_2(state_, next_actions).view(-1)
                target_q = T.min(target_q1, target_q2)

                q_target = self.reward_scale * rewards + self.gamma * (1 - dones) * (target_q -
                                                                                     self.alpha * log_probs.view(-1))

            # ──────────────────────────────────────────
            # Critic Update
            # ──────────────────────────────────────────
            q1 = self.critic_1(state, actions).view(-1)
            q2 = self.critic_2(state, actions).view(-1)

            critic_1_loss = F.mse_loss(q1, q_target)
            critic_2_loss = F.mse_loss(q2, q_target)

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            (critic_1_loss + critic_2_loss).backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # ──────────────────────────────────────────
            # Actor Update
            # ──────────────────────────────────────────
            mu, sigma = self.actor(state)
            dist = Normal(mu, sigma)
            sampled_actions, log_probs = self._sample_action_and_log_prob(dist)

            q1_new = self.critic_1(state, sampled_actions)
            q2_new = self.critic_2(state, sampled_actions)
            q_new = T.min(q1_new, q2_new).view(-1)

            actor_loss = (self.alpha * log_probs.view(-1) - q_new).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ──────────────────────────────────────────
            # Entropy (Alpha) Update
            # ──────────────────────────────────────────
            if self.entropy_tuning:
                entropy_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                entropy_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp().detach()
            else:
                entropy_loss = T.tensor(0.0)

            # ✔️ Track losses for logging in Trainer
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2
            self.last_ent_coef_loss = entropy_loss.item() if self.entropy_tuning else np.nan

            self.learn_step += 1

            self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _sample_action_and_log_prob(self, dist):
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob -= T.log(1 - T.tanh(action).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        action = T.tanh(action) * T.tensor(self.max_action).to(self.device)
        return action, log_prob
