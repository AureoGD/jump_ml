import gymnasium as gym

import numpy as np
from custom_sac.agent import Agent
from torch.utils.tensorboard import SummaryWriter
from custom_sac.agent import Agent
from jump_modular_env.jump_env import JumperEnv  #
import os
from datetime import datetime

# ─────────────────────────────────────────────
# Create log directory
# ─────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
chkpt_dir = f"models/sac_{timestamp}"
os.makedirs(chkpt_dir, exist_ok=True)

env = JumperEnv(policy_type="multihead", discrete_actions=False)
agent = Agent(env=env, chkpt_dir=chkpt_dir, alpha=0.0001)

n_episodes = 1000
max_steps = 2000

writer = agent.writer

reward_history = []
length_history = []
N = 100  # Moving average window

for episode in range(n_episodes):
    observation, _ = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0

    while not done and episode_steps < max_steps:
        action = agent.choose_action(observation)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(observation, action, reward, next_obs, done)
        agent.learn()

        episode_reward += reward
        episode_steps += 1
        observation = next_obs

    # Update reward and length history
    reward_history.append(episode_reward)
    length_history.append(episode_steps)

    # Compute moving averages
    mean_reward = np.mean(reward_history[-N:])
    mean_length = np.mean(length_history[-N:])

    # ─────────────────────────────
    # Log both raw and smoothed
    # ─────────────────────────────
    writer.add_scalar("Episode/Reward", episode_reward, episode)
    writer.add_scalar("Episode/MeanReward", mean_reward, episode)

    writer.add_scalar("Episode/Length", episode_steps, episode)
    writer.add_scalar("Episode/MeanLength", mean_length, episode)

    # Log losses
    agent.log_episode_losses(episode)

    print(f"Episode {episode} | Reward: {episode_reward:.2f} | MeanReward: {mean_reward:.2f} | Steps: {episode_steps}")
