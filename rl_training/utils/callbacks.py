# rl_training/utils/callbacks.py

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveBestModelCallback(BaseCallback):
    """
    Callback to save the model when the mean reward improves.
    """

    def __init__(self, check_freq, save_path, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Every check_freq steps, evaluate
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean(
                    [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                )

                if self.verbose > 0:
                    print(
                        f"[Callback] Step {self.num_timesteps}: Mean reward: {mean_reward:.2f} Best: {self.best_mean_reward:.2f}"
                    )

                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    path = os.path.join(self.save_path, "best_model")
                    self.model.save(path)
                    if self.verbose > 0:
                        print(f"[Callback] Saving new best model to {path}")
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values on TensorBoard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # You can log additional custom metrics here
        # Example: current learning rate, rewards, etc.
        if "lr" in self.model.lr_schedule.__dict__:
            current_lr = self.model.lr_schedule(self.num_timesteps)
            self.logger.record("train/current_learning_rate", current_lr)

        return True
