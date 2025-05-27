import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from datetime import datetime
import os

# ─────────────────────────────────────────────
# Logger setup for TensorBoard
# ─────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/sb3_sac_pendulum_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────
env = Monitor(gym.make("Pendulum-v1"))

# ─────────────────────────────────────────────
# Configure the logger to write to TensorBoard
# ─────────────────────────────────────────────
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# ─────────────────────────────────────────────
# Match the MLP architecture
# ─────────────────────────────────────────────
policy_kwargs = dict(
    net_arch=[256, 256],  # Same as your custom SAC
)

# ─────────────────────────────────────────────
# SAC Model setup
# ─────────────────────────────────────────────
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    tensorboard_log=log_dir,  # Logs also go to TensorBoard automatically
)

model.set_logger(new_logger)

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
model.learn(total_timesteps=1_000_000)

# ─────────────────────────────────────────────
# Save the model
# ─────────────────────────────────────────────
model.save(f"{log_dir}/sac_pendulum")

env.close()
