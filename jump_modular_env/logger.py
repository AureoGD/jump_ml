import os
import pickle
from datetime import datetime


class Logger:
    def __init__(
        self, log_interval=10, data_root="Data", model_save_path=None, model_ref=None
    ):
        self.log_interval = log_interval
        self.enable_logging = False
        self.logged_states = []

        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)

        self.log_folder = self.data_root  # No second timestamp here
        os.makedirs(self.log_folder, exist_ok=True)

        self.ml_file_name = os.path.join(self.log_folder, "ml_data.pkl")
        self.ml_data_list = []

        # Best model saving
        self.best_reward = -float("inf")
        self.model_save_path = model_save_path
        self.model_ref = model_ref  # Pointer to the model to save (agent)

    def update(self, episode, reward_fcns, total_steps):
        if self.enable_logging and self.logged_states:
            self._save_episode_log(episode)

        self.enable_logging = episode % self.log_interval == 0
        self._collect_metrics(reward_fcns)
        self._save_ml_data()
        self.logged_states.clear()
        self.ml_data_list.clear()

        self._check_and_save_best_model(reward_fcns.episode_reward)
        self._maybe_save_checkpoint(total_steps)

    def log_step_state(self, state):
        if self.enable_logging:
            self.logged_states.append(state)

    def _save_episode_log(self, ep):
        file_path = os.path.join(self.log_folder, f"robot-log-ep-{ep}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.logged_states, f)

    def _collect_metrics(self, fcns):
        self.ml_data_list.extend(
            [fcns.episode_reward, fcns.n_jumps, fcns.total_mode_changes, fcns.inter]
        )

    def _save_ml_data(self):
        if os.path.exists(self.ml_file_name):
            with open(self.ml_file_name, "rb") as f:
                try:
                    data_list = pickle.load(f)
                except EOFError:
                    data_list = []
        else:
            data_list = []

        data_list.append(self.ml_data_list)

        with open(self.ml_file_name, "wb") as f:
            pickle.dump(data_list, f)

    def _check_and_save_best_model(self, current_episode_reward):
        if self.model_save_path is not None and self.model_ref is not None:
            if current_episode_reward > self.best_reward:
                self.best_reward = current_episode_reward
                # print(
                #     f"[Logger] Saving new best model! Reward: {current_episode_reward:.2f}"
                # )
                self.model_ref.save(os.path.join(self.model_save_path, "best_model"))

    def save_last_model(self):
        if self.model_save_path is not None and self.model_ref is not None:
            # print(f"[Logger] Saving last model after training.")
            self.model_ref.save(os.path.join(self.model_save_path, "last_model"))

    def _maybe_save_checkpoint(self, total_steps):
        if self.model_save_path is not None and self.model_ref is not None:
            if total_steps > 0 and total_steps % 5000 == 0:
                checkpoint_name = os.path.join(
                    self.model_save_path, f"checkpoint_{total_steps}_steps"
                )
                print(f"[Logger] Saving checkpoint model at {total_steps} steps.")
                self.model_ref.save(checkpoint_name)
