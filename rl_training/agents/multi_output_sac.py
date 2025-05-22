import torch
import torch.nn as nn
from stable_baselines3.sac.sac import SAC


class MultiOutputSAC(SAC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    SAC with auxiliary outputs:
    - success ∈ [0, 1] → MSE Loss
    - stagnation ∈ [0, 1] → BCE Loss
    - phase ∈ {0,1,2,3} → CrossEntropy Loss
    """

    def _setup_model(self):
        super()._setup_model()
        # Ensure the auxiliary heads are created
        if hasattr(self.actor, "_ensure_aux_heads"):
            self.actor._ensure_aux_heads()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        if not hasattr(self, 'ent_coef_tensor'):
            raise RuntimeError("ent_coef_tensor is not initialized! Did _setup_model() run?")
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)

            # Determine entropy coefficient correctly
            if isinstance(self.ent_coef, str) and self.ent_coef == "auto":
                ent_coef = self.ent_coef_tensor
            else:
                ent_coef = torch.as_tensor(self.ent_coef).to(self.device)

            # ---------- Standard SAC Loss ----------
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2) - ent_coef * next_log_prob
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ---------- Actor Loss ----------
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            q1_pi, q2_pi = self.critic(replay_data.observations, actions_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ---------- Entropy Loss ----------
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef_tensor = self.log_ent_coef.exp().detach()
            else:
                ent_coef_loss = torch.tensor(0.0).to(self.device)

            # ---------- Auxiliary Loss ----------
            # self.actor._ensure_aux_heads()

            features = self.actor.extract_features(replay_data.observations)
            pred_success = self.actor.success_head(features).squeeze(-1)
            pred_stagnation = self.actor.stagnation_head(features).squeeze(-1)
            pred_phase = self.actor.phase_head(features)

            success_loss = nn.MSELoss()(pred_success, replay_data.success.squeeze(-1))
            stagnation_loss = nn.BCELoss()(pred_stagnation, replay_data.stagnation.squeeze(-1))
            phase_loss = nn.CrossEntropyLoss()(pred_phase, replay_data.phase.squeeze(-1).long())

            aux_loss = success_loss + stagnation_loss + phase_loss

            self.actor.optimizer.zero_grad()
            aux_loss.backward()
            self.actor.optimizer.step()

            # ---------- Target Update ----------
            self._update_target_network()

            # ---------- Logging ----------
            self.logger.record("train/critic_loss", critic_loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.logger.record("train/entropy_loss", ent_coef_loss.item())
            self.logger.record("train/success_loss", success_loss.item())
            self.logger.record("train/stagnation_loss", stagnation_loss.item())
            self.logger.record("train/phase_loss", phase_loss.item())
