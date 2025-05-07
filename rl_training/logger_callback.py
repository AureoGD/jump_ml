from stable_baselines3.common.callbacks import BaseCallback


class LoggerCallback(BaseCallback):
    def __init__(self, logger, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.custom_logger = logger
        self.check_freq = check_freq
        self.total_steps = 0

    def _on_step(self) -> bool:
        current_steps = self.num_timesteps

        if current_steps > 0 and current_steps % self.check_freq == 0:
            self.custom_logger._maybe_save_checkpoint(current_steps)

        return True

    def _on_rollout_end(self) -> None:
        episode_rewards = self.locals.get("rewards", None)
        if episode_rewards is not None and len(episode_rewards) > 0:
            self.custom_logger.update(
                episode=self.num_timesteps,
                reward_fcns=self.training_env.envs[0].ml_wrapper,
                total_steps=self.total_steps,
            )
