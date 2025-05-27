import os
import time
import numpy as np
from datetime import datetime
from custom_sac.agent import Agent


class Trainer:

    def __init__(
        self,
        env,
        n_episodes=1000,
        max_steps=1000,
        use_encoder=False,
        log_interval=800,
        log_root="runs",
        model_root="models",
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.use_encoder = use_encoder
        self.log_interval = log_interval

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_root, f"sac_{self.env.spec.id}_{timestamp}")
        self.model_dir = os.path.join(model_root, f"sac_{self.env.spec.id}_{timestamp}")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.agent = self.setup_agent()
        self.writer = self.agent.writer

        # Tracking
        self.total_timesteps = 0
        self.total_episodes = 0
        self.n_updates = 0
        self.start_time = time.time()
        self.last_log_timestep = 0

    def setup_agent(self):
        obs_shape = self.env.observation_space.shape
        obs_shapes = (obs_shape[0], ) if not self.use_encoder else self.get_obs_shapes()

        agent = Agent(
            env=self.env,
            obs_shapes=obs_shapes,
            use_encoder=self.use_encoder,
            chkpt_dir=self.model_dir,
            log_dir=self.log_dir,
            alpha="auto",
            beta=3e-4,
            gamma=0.99,
            tau=0.005,
            max_size=1_000_000,
            batch_size=256,
            reward_scale=1,
        )
        return agent

    def get_obs_shapes(self):
        raise NotImplementedError("Define obs_shapes for encoder-based models.")

    def train(self):
        reward_history = []
        length_history = []
        window = 50

        for episode in range(self.n_episodes):
            observation = self.env.reset()[0]
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done and episode_steps < self.max_steps:
                action = self.agent.choose_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                self.agent.remember(observation, action, reward, next_observation, done)
                self.agent.learn()

                episode_reward += reward
                episode_steps += 1
                observation = next_observation

            self.total_timesteps += episode_steps
            self.total_episodes += 1
            self.n_updates = self.agent.learn_step

            reward_history.append(episode_reward)
            length_history.append(episode_steps)

            mean_reward = np.mean(reward_history[-window:])
            mean_length = np.mean(length_history[-window:])

            # ✔️ Log by log_interval (SB3 behavior)
            if (self.total_timesteps - self.last_log_timestep) >= self.log_interval:
                self.writer.add_scalar("rollout/ep_rew_mean", mean_reward, self.total_timesteps)
                self.writer.add_scalar("rollout/ep_len_mean", mean_length, self.total_timesteps)

                self.writer.add_scalar("train/actor_loss", self.agent.last_actor_loss, self.total_timesteps)
                self.writer.add_scalar("train/critic_loss", self.agent.last_critic_loss, self.total_timesteps)
                self.writer.add_scalar("train/ent_coef", self.agent.alpha.item(), self.total_timesteps)
                self.writer.add_scalar("train/ent_coef_loss", self.agent.last_ent_coef_loss, self.total_timesteps)
                self.writer.add_scalar("train/learning_rate", self.agent.actor.optimizer.param_groups[0]['lr'],
                                       self.total_timesteps)

                self.writer.flush()
                self.last_log_timestep = self.total_timesteps

            # Terminal log like SB3
            elapsed_time = time.time() - self.start_time
            fps = int(self.total_timesteps / elapsed_time) if elapsed_time > 0 else 0

            self.print_terminal_log(mean_reward=mean_reward,
                                    mean_length=mean_length,
                                    elapsed_time=int(elapsed_time),
                                    fps=fps)

            if episode % 100 == 0:
                self.save_model()

        self.save_model()

    def print_terminal_log(self, mean_reward, mean_length, elapsed_time, fps):
        actor_loss = getattr(self.agent, "last_actor_loss", np.nan)
        critic_loss = getattr(self.agent, "last_critic_loss", np.nan)
        ent_coef = self.agent.alpha.item() if hasattr(self.agent, "alpha") else np.nan
        ent_coef_loss = getattr(self.agent, "last_ent_coef_loss", np.nan)
        learning_rate = self.agent.actor.optimizer.param_groups[0]['lr']

        print("-" * 43)
        print("| rollout/           |")
        print(f"|    ep_len_mean     | {mean_length:7.0f} |")
        print(f"|    ep_rew_mean     | {mean_reward:7.0f} |")
        print("| time/              |")
        print(f"|    episodes        | {self.total_episodes:7} |")
        print(f"|    fps             | {fps:7} |")
        print(f"|    time_elapsed    | {elapsed_time:7} |")
        print(f"|    total_timesteps | {self.total_timesteps:7} |")
        print("| train/             |")
        print(f"|    actor_loss      | {actor_loss:7.3f} |")
        print(f"|    critic_loss     | {critic_loss:7.3f} |")
        print(f"|    ent_coef        | {ent_coef:7.5f} |")
        print(f"|    ent_coef_loss   | {ent_coef_loss:7.3f} |")
        print(f"|    learning_rate   | {learning_rate:7.6f} |")
        print(f"|    n_updates       | {self.n_updates:7} |")
        print("-" * 43)

    def save_model(self):
        self.agent.save_models()

    def load_model(self):
        self.agent.load_models()

    def close(self):
        self.writer.close()
