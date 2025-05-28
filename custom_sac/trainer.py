import os
import time
import json
import numpy as np
import gymnasium as gym
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent  # Assuming agent.py is in the same custom_sac directory


class BaseCallback:
    """
    Base class for Callbacks.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        # The trainer and agent will be set by the Trainer class
        self.trainer = None  # type: Trainer
        self.agent = None  # type: Agent
        self.logger = None  # type: SummaryWriter

    def _on_training_start(self):
        """Called before the first episode a_t training."""
        pass

    def on_training_start(self):
        self._on_training_start()

    def _on_training_end(self):
        """Called after the training loop is finished."""
        pass

    def on_training_end(self):
        self._on_training_end()

    def _on_episode_start(self):
        """Called at the beginning of each episode."""
        pass

    def on_episode_start(self):
        self._on_episode_start()

    def _on_step(self) -> bool:
        """
        Called after each `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        return self._on_step()

    def _on_rollout_end(self):
        """Called after collecting all steps in an episode, before learning."""
        pass

    def on_rollout_end(self):
        self._on_rollout_end()

    def _on_episode_end(self):
        """Called after the episode loop, including learning updates for that episode's steps."""
        pass

    def on_episode_end(self):
        self._on_episode_end()


class Trainer:

    def __init__(
        self,
        env,
        eval_env=None,
        training_total_timesteps=100000,
        max_steps_per_episode=1000,
        use_encoder=False,
        log_interval_timesteps=2048,  # Changed default to be closer to SB3's typical console log interval
        eval_frequency_timesteps=10000,
        n_eval_episodes=5,
        log_root="runs_custom_sac",
        model_root="models_custom_sac",
        save_freq_episodes=100,  # Can also be timesteps based if preferred
        callback=None,
        # Agent specific HPs, these will be passed to Agent constructor
        gamma=0.99,
        tau=0.005,
        alpha="auto",
        critic_lr=3e-4,
        actor_lr=3e-4,
        replay_buffer_size=1_000_000,
        batch_size=256,
        learning_starts=100,  # Matched SB3's typical default for SAC
        gradient_steps=1,
        reward_scale=1.0,
        agent_kwargs=None,
    ):
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env

        try:
            self.env_name = self.env.spec.id if self.env.spec else "UnknownEnv"
        except AttributeError:
            self.env_name = env.unwrapped.spec.id if hasattr(env, 'unwrapped') and hasattr(
                env.unwrapped, 'spec') and env.unwrapped.spec else "UnknownEnv"

        self.training_total_timesteps = training_total_timesteps
        self.max_steps_per_episode = max_steps_per_episode
        self.use_encoder_flag = use_encoder  # Flag used by _get_obs_shapes_from_env

        self.log_interval_timesteps = log_interval_timesteps
        self.eval_frequency_timesteps = eval_frequency_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.save_freq_episodes = save_freq_episodes

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sac_{self.env_name.replace('/', '-')}_{timestamp}"
        self.log_dir_run = os.path.join(log_root, run_name)
        self.model_dir_run = os.path.join(model_root, run_name)

        os.makedirs(self.log_dir_run, exist_ok=True)
        os.makedirs(self.model_dir_run, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir_run)

        self.trainer_hparams = {
            'training_total_timesteps': training_total_timesteps,
            'max_steps_per_episode': max_steps_per_episode,
            'use_encoder_flag_for_obs_setup': self.use_encoder_flag,
            'log_interval_timesteps': log_interval_timesteps,
            'eval_frequency_timesteps': eval_frequency_timesteps,
            'n_eval_episodes': n_eval_episodes,
            'save_freq_episodes': save_freq_episodes,
            'env_id': self.env_name,
        }
        # Hyperparameters to be passed to the Agent constructor
        self.agent_constructor_hparams = {
            'gamma': gamma,
            'tau': tau,
            'alpha': alpha,
            'critic_lr': critic_lr,
            'actor_lr': actor_lr,
            'max_size': replay_buffer_size,
            'batch_size': batch_size,
            'learning_starts': learning_starts,
            'gradient_steps': gradient_steps,
            'reward_scale': reward_scale,
            'use_encoder': self.use_encoder_flag
            # obs_shapes, chkpt_dir, log_dir are handled by _setup_agent_instance
        }
        if agent_kwargs: self.agent_constructor_hparams.update(agent_kwargs)

        self.agent = self._setup_agent_instance()
        # Handle agent's writer: close agent's self-created writer and make it use Trainer's writer
        if hasattr(self.agent, 'writer') and self.agent.writer is not None:
            if self.agent.writer != self.writer:
                self.agent.writer.close()
            self.agent.writer = self.writer

        self._save_hyperparameters()

        if callback is None: self._callbacks = []
        elif isinstance(callback, list): self._callbacks = callback
        else: self._callbacks = [callback]
        for cb in self._callbacks:
            cb.trainer = self
            cb.agent = self.agent
            cb.logger = self.writer

        self.total_timesteps = 0
        self.total_episodes_completed = 0
        self.start_time = None
        self.last_log_timestep = 0
        self.last_eval_timestep = 0
        self.last_log_time = None
        self.best_mean_eval_reward = -np.inf
        self._train_metric_accumulators = {}
        self._train_metric_counts = {}

    def _save_hyperparameters(self):
        all_hparams = {
            "trainer_hyperparameters": self.trainer_hparams,
            "agent_constructor_hyperparameters": self.agent_constructor_hparams,
            "agent_internal_hyperparameters": self.agent.hparams if hasattr(self.agent, 'hparams') else {}
        }
        filepath = os.path.join(self.model_dir_run, "hyperparameters.json")
        with open(filepath, 'w') as f:
            json.dump(all_hparams, f, indent=4, sort_keys=True)
        # print(f"Hyperparameters saved to {filepath}") # Optional

    def _get_obs_shapes_from_env(self):
        obs_space = self.env.observation_space
        if self.use_encoder_flag:
            if not isinstance(obs_space, gym.spaces.Dict):
                raise ValueError(f"use_encoder=True requires Dict obs space, got {type(obs_space)}")
            obs_shapes_dict = {}
            for key, space in obs_space.spaces.items():
                if not isinstance(space, gym.spaces.Box):
                    raise ValueError(f"Encoder input for key '{key}' must be Box, got {type(space)}")
                obs_shapes_dict[key] = space.shape
            if not obs_shapes_dict: raise ValueError("Obs space dict is empty with use_encoder=True.")
            return obs_shapes_dict
        else:
            if not isinstance(obs_space, gym.spaces.Box):
                raise ValueError(f"use_encoder=False expects Box obs space, got {type(obs_space)}")
            return obs_space.shape

    def _setup_agent_instance(self):
        current_obs_shapes = self._get_obs_shapes_from_env()
        agent_instance = Agent(
            env=self.env,
            obs_shapes=current_obs_shapes,
            # use_encoder is passed directly from agent_constructor_hparams
            chkpt_dir=self.model_dir_run,
            log_dir=self.log_dir_run,  # Agent creates writer here, Trainer will manage it
            **self.agent_constructor_hparams  # Unpack all other agent hparams
        )
        return agent_instance

    def _call_callbacks(self, event_method_name):
        continue_training = True
        for callback in self._callbacks:
            method = getattr(callback, event_method_name, None)
            if method and not method(): continue_training = False
        return continue_training

    def _reset_train_metric_accumulators(self):
        self._train_metric_accumulators = {"actor_loss": 0.0, "critic_loss": 0.0, "ent_coef_loss": 0.0}
        self._train_metric_counts = {"actor_loss": 0, "critic_loss": 0, "ent_coef_loss": 0}

    def _accumulate_train_metrics(self):
        # This is called after agent.learn() which performs agent.gradient_steps internally
        num_gradient_steps_in_last_learn_call = self.agent.gradient_steps

        if not np.isnan(self.agent.last_actor_loss):
            self._train_metric_accumulators[
                "actor_loss"] += self.agent.last_actor_loss * num_gradient_steps_in_last_learn_call
            self._train_metric_counts["actor_loss"] += num_gradient_steps_in_last_learn_call
        if not np.isnan(self.agent.last_critic_loss):
            self._train_metric_accumulators[
                "critic_loss"] += self.agent.last_critic_loss * num_gradient_steps_in_last_learn_call
            self._train_metric_counts["critic_loss"] += num_gradient_steps_in_last_learn_call
        if self.agent.entropy_tuning and not np.isnan(self.agent.last_ent_coef_loss):
            self._train_metric_accumulators[
                "ent_coef_loss"] += self.agent.last_ent_coef_loss * num_gradient_steps_in_last_learn_call
            self._train_metric_counts["ent_coef_loss"] += num_gradient_steps_in_last_learn_call

    def _run_evaluation(self):
        if self.n_eval_episodes <= 0: return
        print(f"\nRunning evaluation at timestep {self.total_timesteps}...")
        eval_rewards, eval_lengths = [], []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done, ep_rew, ep_len = False, 0, 0
            # Limit eval episode steps to avoid infinite loops in problematic envs
            for _eval_step in range(self.max_steps_per_episode):
                if done: break
                action = self.agent.choose_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                ep_rew += reward
                ep_len += 1
                obs = next_obs
            eval_rewards.append(ep_rew)
            eval_lengths.append(ep_len)

        mean_reward, mean_length = np.mean(eval_rewards), np.mean(eval_lengths)
        self.writer.add_scalar("eval/mean_reward", mean_reward, self.total_timesteps)
        self.writer.add_scalar("eval/mean_episode_length", mean_length, self.total_timesteps)
        self.writer.flush()

        if mean_reward > self.best_mean_eval_reward:
            print(
                f"Evaluation: New best eval reward: {mean_reward:.2f} (old: {self.best_mean_eval_reward:.2f}). Saving best model..."
            )
            self.best_mean_eval_reward = mean_reward
            self.agent.save_models(best_model=True)
        else:
            print(f"Evaluation: Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}")
        self.last_eval_timestep = self.total_timesteps

    def _log_terminal_summary(self, mean_rollout_reward, mean_rollout_length, interval_fps, total_time_elapsed):
        LABEL_WIDTH, VALUE_WIDTH, TOTAL_WIDTH = 26, 15, 50  # Adjusted total width

        actor_loss_val, critic_loss_val, ent_coef_loss_val = np.nan, np.nan, np.nan
        if self._train_metric_counts.get("actor_loss", 0) > 0:
            actor_loss_val = self._train_metric_accumulators["actor_loss"] / self._train_metric_counts["actor_loss"]
        if self._train_metric_counts.get("critic_loss", 0) > 0:
            critic_loss_val = self._train_metric_accumulators["critic_loss"] / self._train_metric_counts["critic_loss"]
        if self.agent.entropy_tuning and self._train_metric_counts.get("ent_coef_loss", 0) > 0:
            ent_coef_loss_val = self._train_metric_accumulators["ent_coef_loss"] / self._train_metric_counts[
                "ent_coef_loss"]

        ent_coef_val = self.agent.alpha.item()
        actor_lr_val = self.agent.actor.optimizer.param_groups[0][
            'lr'] if self.agent.actor.optimizer.param_groups else np.nan
        n_updates_val = self.agent.learn_step_counter

        print("\n" + "-" * TOTAL_WIDTH)
        print(f"| {f'Section/Metric':<{LABEL_WIDTH}} | {f'Value':>{VALUE_WIDTH}} |")
        print(f"|{'':-<{LABEL_WIDTH+1}}|{'':-<{VALUE_WIDTH+2}}|")
        print(f"| {f'rollout/':<{LABEL_WIDTH}} | {'':>{VALUE_WIDTH}} |")
        print(f"| {f'  ep_len_mean':<{LABEL_WIDTH}} | {mean_rollout_length:>{VALUE_WIDTH}.2f} |")
        print(f"| {f'  ep_rew_mean':<{LABEL_WIDTH}} | {mean_rollout_reward:>{VALUE_WIDTH}.2e} |")
        print(f"|{'':-<{LABEL_WIDTH+1}}|{'':-<{VALUE_WIDTH+2}}|")
        print(f"| {f'time/':<{LABEL_WIDTH}} | {'':>{VALUE_WIDTH}} |")
        print(f"| {f'  episodes':<{LABEL_WIDTH}} | {self.total_episodes_completed:>{VALUE_WIDTH}d} |")
        print(f"| {f'  fps':<{LABEL_WIDTH}} | {interval_fps:>{VALUE_WIDTH}d} |")
        print(f"| {f'  time_elapsed':<{LABEL_WIDTH}} | {int(total_time_elapsed):>{VALUE_WIDTH}d} |")
        print(f"| {f'  total_timesteps':<{LABEL_WIDTH}} | {self.total_timesteps:>{VALUE_WIDTH}d} |")
        print(f"|{'':-<{LABEL_WIDTH+1}}|{'':-<{VALUE_WIDTH+2}}|")
        print(f"| {f'train/':<{LABEL_WIDTH}} | {'':>{VALUE_WIDTH}} |")
        print(f"| {f'  actor_loss':<{LABEL_WIDTH}} | {actor_loss_val:>{VALUE_WIDTH}.3f} |")
        print(f"| {f'  critic_loss':<{LABEL_WIDTH}} | {critic_loss_val:>{VALUE_WIDTH}.3f} |")
        print(f"| {f'  ent_coef':<{LABEL_WIDTH}} | {ent_coef_val:>{VALUE_WIDTH}.5f} |")
        print(f"| {f'  ent_coef_loss':<{LABEL_WIDTH}} | {ent_coef_loss_val:>{VALUE_WIDTH}.3f} |")
        print(f"| {f'  learning_rate':<{LABEL_WIDTH}} | {actor_lr_val:>{VALUE_WIDTH}.2e} |")
        print(f"| {f'  n_updates':<{LABEL_WIDTH}} | {n_updates_val:>{VALUE_WIDTH}d} |")
        print("-" * TOTAL_WIDTH)

    def train(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_timestep = 0
        self.last_eval_timestep = 0
        if not self._call_callbacks('on_training_start'): return
        self._reset_train_metric_accumulators()
        reward_history_window, length_history_window, current_episode_num = [], [], 0
        window_size = 100

        while self.total_timesteps < self.training_total_timesteps:
            current_episode_num += 1
            self.total_episodes_completed = current_episode_num
            if not self._call_callbacks('on_episode_start'): break
            observation, _ = self.env.reset()
            done, episode_reward, episode_steps = False, 0, 0

            while not done and episode_steps < self.max_steps_per_episode and self.total_timesteps < self.training_total_timesteps:
                action = self.agent.choose_action(observation, evaluate=False)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.total_timesteps += 1
                episode_steps += 1
                episode_reward += reward
                self.agent.remember(observation, action, reward, next_observation, done)

                if self.total_timesteps >= self.agent.learning_starts:
                    if self.agent.learn(): self._accumulate_train_metrics()

                if not self._call_callbacks('on_step'):
                    self.total_timesteps = self.training_total_timesteps
                    done = True
                observation = next_observation

            if not self._call_callbacks('on_rollout_end'): break

            reward_history_window.append(episode_reward)
            length_history_window.append(episode_steps)
            if len(reward_history_window) > window_size: reward_history_window.pop(0)
            if len(length_history_window) > window_size: length_history_window.pop(0)
            mean_reward_w = np.mean(reward_history_window) if reward_history_window else np.nan
            mean_length_w = np.mean(length_history_window) if length_history_window else np.nan

            self.writer.add_scalar("rollout/ep_reward", episode_reward, self.total_timesteps)
            self.writer.add_scalar("rollout/ep_length", episode_steps, self.total_timesteps)
            self.writer.add_scalar("rollout/ep_rew_mean", mean_reward_w, self.total_timesteps)
            self.writer.add_scalar("rollout/ep_len_mean", mean_length_w, self.total_timesteps)

            if (self.total_timesteps -
                    self.last_log_timestep) >= self.log_interval_timesteps and self.total_timesteps > 0:
                curr_time = time.time()
                total_time_elapsed = curr_time - self.start_time
                time_for_interval = curr_time - self.last_log_time
                steps_in_interval = self.total_timesteps - self.last_log_timestep
                interval_fps = int(steps_in_interval /
                                   time_for_interval) if time_for_interval > 0 and steps_in_interval > 0 else 0

                self.writer.add_scalar("time/fps", interval_fps, self.total_timesteps)
                self.writer.add_scalar("time/episodes", self.total_episodes_completed, self.total_timesteps)
                self.writer.add_scalar("time/time_elapsed", total_time_elapsed, self.total_timesteps)

                if self._train_metric_counts.get("actor_loss", 0) > 0:
                    self.writer.add_scalar(
                        "train/actor_loss",
                        self._train_metric_accumulators["actor_loss"] / self._train_metric_counts["actor_loss"],
                        self.total_timesteps)
                    self.writer.add_scalar(
                        "train/critic_loss",
                        self._train_metric_accumulators["critic_loss"] / self._train_metric_counts["critic_loss"],
                        self.total_timesteps)
                    if self.agent.entropy_tuning and self._train_metric_counts["ent_coef_loss"] > 0:
                        self.writer.add_scalar(
                            "train/ent_coef_loss", self._train_metric_accumulators["ent_coef_loss"] /
                            self._train_metric_counts["ent_coef_loss"], self.total_timesteps)
                self.writer.add_scalar("train/ent_coef", self.agent.alpha.item(), self.total_timesteps)
                if self.agent.actor.optimizer.param_groups:
                    self.writer.add_scalar("train/learning_rate", self.agent.actor.optimizer.param_groups[0]['lr'],
                                           self.total_timesteps)
                self.writer.add_scalar("train/n_updates", self.agent.learn_step_counter, self.total_timesteps)
                self.writer.flush()

                self._log_terminal_summary(mean_reward_w, mean_length_w, interval_fps, total_time_elapsed)

                self.last_log_timestep = self.total_timesteps
                self.last_log_time = curr_time
                self._reset_train_metric_accumulators()

            # Simplified print per episode if desired (can be commented out if terminal summary is preferred)
            # print(f"Ep: {current_episode_num} | TS: {self.total_timesteps}/{self.training_total_timesteps} | ER: {episode_reward:.1f} ...")

            if self.total_timesteps - self.last_eval_timestep >= self.eval_frequency_timesteps and self.total_timesteps > 0:
                self._run_evaluation()
            if current_episode_num > 0 and current_episode_num % self.save_freq_episodes == 0:
                self.agent.save_models(best_model=False)
            if not self._call_callbacks('on_episode_end'): break
            if self.total_timesteps >= self.training_total_timesteps and done: break

        if not self._call_callbacks('on_training_end'): pass
        self.close()

    def close(self):
        if self.writer: self.writer.close()
        if self.eval_env is not None and self.eval_env != self.env and hasattr(self.eval_env, 'close'):
            self.eval_env.close()
        if hasattr(self.env, 'close'): self.env.close()
