import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import time
from torch.utils.tensorboard import SummaryWriter
import shutil  # for optional cleanup
import os

from jump_modular_env.jump_env import JumperEnv
from logger import MPCLogger  # Make sure logger is in validate_mpc/

# ─────────────────────
# Setup
# ─────────────────────
mode = 2
episodes = 1
max_steps = 1500  # Total steps expected per episode

episode_rewards = []
com_proj = []

env = JumperEnv(
    render=True,
    render_every=True,
    render_interval=10,
    log_interval=10,
    debug=True,
    disturb=False,
)

# ─────────────────────
# Evaluation Loop
# ─────────────────────
for episode in range(episodes):
    # Define per-episode log directories (no timestamp)
    controller_name = f"rgc_{mode}_episode_{episode}"
    log_dir_tb = f"validate_mpc/tensorboard/{controller_name}"
    log_dir_csv = f"validate_mpc/mpc_data/{controller_name}"

    # Create writers
    writer = SummaryWriter(log_dir=log_dir_tb)
    logger = MPCLogger(log_dir=log_dir_csv)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    # Storage for plotting CoM, heel, and toe X positions
    com_x_vals = []
    heel_x_vals = []
    toe_x_vals = []

    ic(f"Episode {episode} begins")

    while not done and step_count < max_steps:
        # Mode is fixed, PPO not used
        action = np.array([mode])
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Extract robot states
        rs = env.robot_states

        # Store values for plotting
        com_x_vals.append(rs.r_pos[0])
        heel_x_vals.append(rs.heel_pos[0])
        toe_x_vals.append(rs.toe_pos[0])

        # ────────────────
        # Log to TensorBoard
        # ────────────────
        for i, q in enumerate(rs.q):
            q_val = rs.q[i, 0]
            qr_val = rs.qr[i, 0]

            writer.add_scalar(f"joint_tracking/q_{i}", q_val, step_count)
            writer.add_scalar(f"joint_tracking/qr_{i}", qr_val, step_count)

        for i, dq in enumerate(rs.dq):
            writer.add_scalar(f"joint/dq_{i}", dq, step_count)
            writer.add_scalar(f"joint/tau_{i}", rs.tau[i, 0], step_count)

        writer.add_scalar("com/x", rs.r_pos[0], step_count)
        writer.add_scalar("com/z", rs.r_pos[1], step_count)

        writer.add_scalar("com/vx", rs.r_vel[0], step_count)
        writer.add_scalar("com/vz", rs.r_vel[1], step_count)

        writer.add_scalar("body/theta_y", rs.th[0], step_count)
        writer.add_scalar("body/dtheta_y", rs.dth[0], step_count)
        writer.add_scalar("body/x", rs.b_pos[0], step_count)
        writer.add_scalar("body/z", rs.b_pos[1], step_count)

        # ────────────────
        # Log to CSV
        # ────────────────
        logger.log(step_count, rs)

        step_count += 1

    # ────────────────
    # Save logs for this episode
    # ────────────────
    logger.save()
    writer.close()
    episode_rewards.append(total_reward)
    com_proj.append((com_x_vals, heel_x_vals, toe_x_vals))
    print(f"Valid episode ({step_count} steps) — logs saved to {controller_name}")

env.close()

# ─────────────────────
# Plotting Episode Reward
# ─────────────────────
if episode_rewards:
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, marker="o")
    plt.title(f"Episode Rewards for RGC Mode {mode}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid episodes to plot.")

# ─────────────────────
# Plotting X-Position of CoM, Heel, and Toe (per episode)
# ─────────────────────
if com_proj:
    for i, (com_x, heel_x, toe_x) in enumerate(com_proj):
        plt.figure(figsize=(10, 5))
        plt.plot(com_x, label="CoM X", linewidth=2)
        plt.plot(heel_x, label="Heel X", linestyle="--")
        plt.plot(toe_x, label="Toe X", linestyle="-.")

        plt.title(f"[Episode {i}] X-Position of CoM, Heel, and Toe — RGC Mode {mode}")
        plt.xlabel("Time Step")
        plt.ylabel("X Position (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
else:
    print("No CoM/foot positions to plot.")
