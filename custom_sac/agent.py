import os
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from .networks import ActorNetwork, CriticNetwork
from .replay_buffer import ReplayBuffer


class Agent:

    def __init__(
        self,
        env,
        obs_shapes,
        use_encoder=True,
        alpha="auto",
        critic_lr=3e-4,
        actor_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        max_size=1_000_000,
        batch_size=256,
        reward_scale=1.0,
        chkpt_dir="tmp/sac",
        log_dir="runs/sac",
        gradient_steps=1,
        learning_starts=1000,
        policy_delay=2,  # How often to update policy and target networks
        max_grad_norm=None,
        # Max norm for gradient clipping (e.g., 1.0). None to disable.
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.policy_delay = policy_delay
        self.max_grad_norm = max_grad_norm  # Store max_grad_norm

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.obs_shapes = obs_shapes
        self.use_encoder = use_encoder

        self.writer = SummaryWriter(log_dir=log_dir)
        self.learn_step_counter = 0

        self.is_dict_obs = isinstance(obs_shapes, dict) and use_encoder
        self.n_actions = env.action_space.shape[0]

        self.max_action = T.as_tensor(env.action_space.high, dtype=T.float32, device=self.device)
        if self.max_action.ndim == 0:
            self.max_action = self.max_action.unsqueeze(0)
        if self.max_action.shape[0] != self.n_actions and self.max_action.shape[0] == 1:
            self.max_action = self.max_action.repeat(self.n_actions)
        elif self.max_action.shape[0] != self.n_actions:
            raise ValueError(f"max_action shape {self.max_action.shape} incompatible with n_actions {self.n_actions}")

        actor_net_max_action_scalar = float(env.action_space.high[0])
        if np.isscalar(env.action_space.high):
            actor_net_max_action_scalar = float(env.action_space.high)
        else:
            actor_net_max_action_scalar = float(env.action_space.high[0])

        self.hparams = {
            'agent_type': 'SAC_Custom',
            'obs_shapes_str': str(obs_shapes),
            'use_encoder': use_encoder,
            'is_dict_obs': self.is_dict_obs,
            'alpha_init': alpha,
            'critic_lr': critic_lr,
            'actor_lr': actor_lr,
            'gamma': gamma,
            'tau': tau,
            'replay_buffer_max_size': max_size,
            'batch_size': batch_size,
            'reward_scale': reward_scale,
            'gradient_steps': gradient_steps,
            'learning_starts': learning_starts,
            'policy_delay': policy_delay,
            'max_grad_norm': max_grad_norm,  # Added to hparams
            'n_actions': self.n_actions,
            'actor_net_max_action_scalar': actor_net_max_action_scalar
        }

        self.actor = ActorNetwork(alpha=actor_lr,
                                  obs_shapes=obs_shapes,
                                  max_action=actor_net_max_action_scalar,
                                  n_actions=self.n_actions,
                                  chkpt_dir=chkpt_dir,
                                  use_encoder=use_encoder,
                                  name="actor")
        self.critic_1 = CriticNetwork(beta=critic_lr,
                                      obs_shapes=obs_shapes,
                                      n_actions=self.n_actions,
                                      chkpt_dir=chkpt_dir,
                                      name="critic_1",
                                      use_encoder=use_encoder)
        self.critic_2 = CriticNetwork(beta=critic_lr,
                                      obs_shapes=obs_shapes,
                                      n_actions=self.n_actions,
                                      chkpt_dir=chkpt_dir,
                                      name="critic_2",
                                      use_encoder=use_encoder)
        self.target_critic_1 = CriticNetwork(beta=critic_lr,
                                             obs_shapes=obs_shapes,
                                             n_actions=self.n_actions,
                                             chkpt_dir=chkpt_dir,
                                             name="target_critic_1",
                                             use_encoder=use_encoder)
        self.target_critic_2 = CriticNetwork(beta=critic_lr,
                                             obs_shapes=obs_shapes,
                                             n_actions=self.n_actions,
                                             chkpt_dir=chkpt_dir,
                                             name="target_critic_2",
                                             use_encoder=use_encoder)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        replay_buffer_obs_shapes = obs_shapes if self.is_dict_obs else {"obs": obs_shapes}
        self.memory = ReplayBuffer(
            max_size=max_size,
            obs_space_dict=replay_buffer_obs_shapes,
            n_actions=self.n_actions,
        )

        if isinstance(alpha, str) and alpha.lower() == "auto":
            self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32).item()
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=critic_lr)
            self.entropy_tuning = True
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.entropy_tuning = False
            self.alpha = T.tensor(float(alpha), device=self.device)

        self.last_actor_loss, self.last_critic_loss, self.last_ent_coef_loss = np.nan, np.nan, np.nan

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()
        if self.is_dict_obs:
            obs_tensor = {
                k: T.tensor(np.array(v), dtype=T.float32).unsqueeze(0).to(self.device)
                for k, v in observation.items()
            }
        else:
            obs_tensor = T.tensor(np.array(observation), dtype=T.float32).unsqueeze(0).to(self.device)
        with T.no_grad():
            pre_tanh_mu, sigma = self.actor.forward(obs_tensor)
            action_gaussian = pre_tanh_mu if evaluate else Normal(pre_tanh_mu, sigma).rsample()
            action_tanh = T.tanh(action_gaussian)
        scaled_action = action_tanh * self.max_action
        self.actor.train()
        return scaled_action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        state_to_store = state if self.is_dict_obs else {"obs": state}
        new_state_to_store = new_state if self.is_dict_obs else {"obs": new_state}
        self.memory.store_transition(state_to_store, action, reward, new_state_to_store, done)

    def save_models(self, best_model=False):
        print(f"... saving {'best ' if best_model else ''}models ...")
        suffix = "_best" if best_model else ""
        self.actor.save_checkpoint(suffix=suffix)
        self.critic_1.save_checkpoint(suffix=suffix)
        self.critic_2.save_checkpoint(suffix=suffix)

    def load_models(self, best_model=False):
        print(f"... loading {'best ' if best_model else ''}models ...")
        suffix = "_best" if best_model else ""
        self.actor.load_checkpoint(suffix=suffix)
        self.critic_1.load_checkpoint(suffix=suffix)
        self.critic_2.load_checkpoint(suffix=suffix)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def learn(self):
        if self.memory.size() < max(self.learning_starts, self.batch_size): return False

        for i in range(self.gradient_steps):
            states_data, actions_data, rewards_data, next_states_data, dones_data = \
                self.memory.sample_buffer(self.batch_size)
            if self.is_dict_obs:
                states = {k: T.tensor(states_data[k], dtype=T.float32).to(self.device) for k in states_data.keys()}
                next_states = {
                    k: T.tensor(next_states_data[k], dtype=T.float32).to(self.device)
                    for k in next_states_data.keys()
                }
            else:
                states = T.tensor(states_data["obs"], dtype=T.float32).to(self.device)
                next_states = T.tensor(next_states_data["obs"], dtype=T.float32).to(self.device)
            actions = T.tensor(actions_data, dtype=T.float32).to(self.device)
            rewards = T.tensor(rewards_data, dtype=T.float32).to(self.device).unsqueeze(1)
            dones = T.tensor(dones_data, dtype=T.float32).to(self.device).unsqueeze(1)

            with T.no_grad():
                next_policy_actions_scaled, next_log_probs = self.actor.sample_normal(next_states, reparameterize=False)
                target_q1 = self.target_critic_1.forward(next_states, next_policy_actions_scaled)
                target_q2 = self.target_critic_2.forward(next_states, next_policy_actions_scaled)
                target_q_min = T.min(target_q1, target_q2)
                q_target = self.reward_scale * rewards + self.gamma * (1.0 - dones) * (target_q_min -
                                                                                       self.alpha * next_log_probs)

            q1_current = self.critic_1.forward(states, actions)
            q2_current = self.critic_2.forward(states, actions)
            critic_1_loss = F.mse_loss(q1_current, q_target)
            critic_2_loss = F.mse_loss(q2_current, q_target)
            critic_loss_total = critic_1_loss + critic_2_loss

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            critic_loss_total.backward()
            # *** GRADIENT CLIPPING FOR CRITICS ***
            if self.max_grad_norm is not None:
                T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
                T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()
            self.last_critic_loss = critic_loss_total.item() / 2.0

            self.learn_step_counter += 1

            if self.learn_step_counter % self.policy_delay == 0:
                for p in self.critic_1.parameters():
                    p.requires_grad = False
                for p in self.critic_2.parameters():
                    p.requires_grad = False

                policy_actions_scaled, log_probs = self.actor.sample_normal(states, reparameterize=True)
                q1_policy = self.critic_1.forward(states, policy_actions_scaled)
                q2_policy = self.critic_2.forward(states, policy_actions_scaled)
                q_policy_min = T.min(q1_policy, q2_policy)
                actor_loss = (self.alpha * log_probs - q_policy_min).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                # *** GRADIENT CLIPPING FOR ACTOR ***
                if self.max_grad_norm is not None:
                    T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.last_actor_loss = actor_loss.item()

                for p in self.critic_1.parameters():
                    p.requires_grad = True
                for p in self.critic_2.parameters():
                    p.requires_grad = True

                if self.entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    # Optionally clip alpha gradient, though less common:
                    # if self.max_grad_norm is not None:
                    #     T.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().detach()
                    self.last_ent_coef_loss = alpha_loss.item()
                else:
                    self.last_ent_coef_loss = np.nan

                self.update_network_parameters()
        return True

    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        for tp, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for tp, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
