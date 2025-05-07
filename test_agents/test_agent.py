# test_agents/test_agent_with_plot.py

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from jump_modular_env.jump_env import JumperEnv

# -------------------------------
# Fixed configuration (no argparse)
# -------------------------------

ALGO = "sac"  # "ppo" or "sac"
POLICY = "cnn"  # "cnn" or "mlp"
MODEL_PATH = "models/SAC/cnn/runs/20250501_162304/models/best_model.zip"  # Update path
N_EPISODES = 1  # Number of episodes to test

# -------------------------------
# Helper functions
# -------------------------------


def load_model(algo, model_path):
    if algo.lower() == "ppo":
        model = PPO.load(model_path)
    elif algo.lower() == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    return model


def plot_mode_history(mode_history, episode_idx, total_reward):
    plt.figure(figsize=(10, 4))
    plt.plot(mode_history, marker="o")
    plt.title(
        f"Mode History - Episode {episode_idx + 1} (Total Reward: {total_reward:.2f})"
    )
    plt.xlabel("Timestep")
    plt.ylabel("Selected Mode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cumulative_reward(reward_list, episode_idx, total_reward):
    cumulative_rewards = np.cumsum(reward_list)
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_rewards, marker="o")
    plt.title(
        f"Cumulative Reward - Episode {episode_idx + 1} (Total Reward: {total_reward:.2f})"
    )
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main execution
# -------------------------------


def main():
    env = JumperEnv(
        render=True, policy_type=POLICY, discrete_actions=(ALGO.lower() == "ppo")
    )

    print(f"Loading model from {MODEL_PATH}")
    model = load_model(ALGO, MODEL_PATH)

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        mode_history = []
        reward_list = []
        inter = 0
        while not done:
            # force_cm = 500

            # while inter < 1000:
            action, _states = model.predict(obs, deterministic=True)

            if isinstance(action, np.ndarray) or isinstance(action, list):
                mode_selected = int(np.round(action[0]))
            else:
                mode_selected = int(np.round(action))
            mode_history.append(mode_selected)
            # if inter < force_cm:
            #     action = 0
            # else:
            #     action = 2
            # mode_history.append(action)
            # inter += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            reward_list.append(reward)

            done = terminated or truncated

        # After episode ends
        print(f"\nEpisode {ep + 1}/{N_EPISODES}")
        print(f"Total reward: {total_reward:.2f}")

        plot_mode_history(mode_history, ep, total_reward)
        plot_cumulative_reward(reward_list, ep, total_reward)


if __name__ == "__main__":
    main()
