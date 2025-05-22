import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer
from typing import NamedTuple


class MultiOutputReplayBufferSamples(NamedTuple):
    observations: dict
    actions: torch.Tensor
    next_observations: dict
    dones: torch.Tensor
    rewards: torch.Tensor
    success: torch.Tensor
    stagnation: torch.Tensor
    phase: torch.Tensor


class MultiOutputReplayBuffer(DictReplayBuffer):
    """
    Replay buffer extended with auxiliary outputs:
    - success ∈ [0, 1] (regression)
    - stagnation ∈ [0, 1] (regression)
    - phase ∈ integer class {0,1,2,3} (classification)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.success = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.stagnation = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.phase = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        super().add(obs, next_obs, action, reward, done, infos)

        for env_idx in range(self.n_envs):
            self.success[self.pos, env_idx] = infos[env_idx].get("success", 0.0)
            self.stagnation[self.pos, env_idx] = infos[env_idx].get("stagnation", 0.0)
            self.phase[self.pos, env_idx] = infos[env_idx].get("phase", 0.0)

    def sample(self, batch_size: int) -> MultiOutputReplayBufferSamples:
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        buffer_indices = np.random.randint(0, self.size(), size=batch_size)

        data = self._custom_get_samples(buffer_indices, env_indices)

        success = torch.tensor(self.success[buffer_indices, env_indices], device=self.device, dtype=torch.float32)
        stagnation = torch.tensor(self.stagnation[buffer_indices, env_indices], device=self.device, dtype=torch.float32)
        phase = torch.tensor(self.phase[buffer_indices, env_indices], device=self.device, dtype=torch.float32)

        return MultiOutputReplayBufferSamples(
            observations=data["observations"],
            actions=data["actions"],
            next_observations=data["next_observations"],
            dones=data["dones"],
            rewards=data["rewards"],
            success=success,
            stagnation=stagnation,
            phase=phase,
        )

    def _custom_get_samples(self, batch_inds, env_inds):
        obs = {key: self.observations[key][batch_inds, env_inds, :] for key in self.observations.keys()}
        next_obs = {key: self.next_observations[key][batch_inds, env_inds, :] for key in self.next_observations.keys()}

        return {
            "observations": {
                key: torch.tensor(value, device=self.device, dtype=torch.float32)
                for key, value in obs.items()
            },
            "next_observations": {
                key: torch.tensor(value, device=self.device, dtype=torch.float32)
                for key, value in next_obs.items()
            },
            "actions": torch.tensor(self.actions[batch_inds, env_inds], device=self.device, dtype=torch.float32),
            "rewards": torch.tensor(self.rewards[batch_inds, env_inds], device=self.device, dtype=torch.float32),
            "dones": torch.tensor(self.dones[batch_inds, env_inds], device=self.device, dtype=torch.float32),
        }
