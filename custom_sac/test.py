import gymnasium as gym
from datetime import datetime
import os

from custom_sac.agent import Agent  # ✅ Assuming agent.py is in custom_sac
from jump_modular_env.jump_env import JumperEnv  # ✅ Your custom env

# ─────────────────────────────────────────────
# Create log directory
# ─────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
chkpt_dir = f"models/sac_{timestamp}"
os.makedirs(chkpt_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Create environment
# ─────────────────────────────────────────────
env = JumperEnv(policy_type="multihead", discrete_actions=False)

# ─────────────────────────────────────────────
# Create agent
# ─────────────────────────────────────────────
agent = Agent(env=env, chkpt_dir=chkpt_dir, alpha=0.0001)

# ─────────────────────────────────────────────
# Test interaction
# ─────────────────────────────────────────────
n_episodes = 10

for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = agent.choose_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(obs, action, reward, next_obs, done)
        agent.learn()

        obs = next_obs
        ep_reward += reward

    print(f"Episode {ep + 1}: Reward = {ep_reward}")

env.close()
