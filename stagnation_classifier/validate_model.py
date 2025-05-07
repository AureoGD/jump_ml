import os
import pickle
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from jump_modular_env.jump_env import JumperEnv
from stagnation_classifier.cnn_stagnation_wrapper import StagnationClassifier

# --- Configuration ---
MAX_STEPS = 3000
TRIM_STEPS = 25  # Trim for plotting
MODEL_PATH = "stagnation_classifier/models/cnn_classifier_IV.pth"

# --- Initialize ---
env = JumperEnv(render=True, disturb=False, policy_type="cnn")
classifier = StagnationClassifier(MODEL_PATH)

# --- User-defined transition logic ---
transition_step = 1500
mode_1 = 0
mode_2 = 2

obs, _ = env.reset()
episode_log = []
stagnation_preds = []
stagnation_intervals = []

# --- Start simulation with video recording ---
video_path = "stagnation_classifier/tmp_episode.mp4"
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)

is_stagnated = False
stagnation_start = None

for step in range(MAX_STEPS):
    action = [mode_1 if step < transition_step else mode_2]
    obs, reward, terminated, truncated, info = env.step(action)

    # Observation windows for classifier
    window = {
        "predictable": np.stack(list(env.ml_wrapper.predictable_history), axis=1),
        "nonpredictable": np.stack(list(env.ml_wrapper.nonpredictable_history), axis=1),
    }
    episode_log.append(window)

    # Get stagnation prediction
    pred = classifier.predict(
        env.ml_wrapper.predictable_history, env.ml_wrapper.nonpredictable_history
    )
    stagnation_preds.append(pred)

    if pred == 1 and not is_stagnated:
        stagnation_start = step
        is_stagnated = True
    elif pred == 0 and is_stagnated:
        stagnation_intervals.append((stagnation_start, step))
        is_stagnated = False

    if terminated or truncated or step == MAX_STEPS - 1:
        if is_stagnated:
            stagnation_intervals.append((stagnation_start, step))
        break

p.stopStateLogging(log_id)

# --- Plotting ---
timesteps = np.arange(len(episode_log[TRIM_STEPS:]))

pos_x = [step["nonpredictable"][9, -1] for step in episode_log[TRIM_STEPS:]]
pos_z = [step["nonpredictable"][10, -1] for step in episode_log[TRIM_STEPS:]]
vel_x = [step["nonpredictable"][7, -1] for step in episode_log[TRIM_STEPS:]]
vel_z = [step["nonpredictable"][8, -1] for step in episode_log[TRIM_STEPS:]]
th_y = [step["predictable"][4, -1] for step in episode_log[TRIM_STEPS:]]
dth_y = [step["predictable"][5, -1] for step in episode_log[TRIM_STEPS:]]

fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()

signals = [
    (pos_x, "Base Position X"),
    (vel_x, "Base Velocity X"),
    (pos_z, "Base Position Z"),
    (vel_z, "Base Velocity Z"),
    (th_y, "Base Orientation (Yaw)"),
    (dth_y, "Base Angular Velocity (Yaw Rate)"),
]

for i, (signal, title) in enumerate(signals):
    axs[i].plot(timesteps, signal, color="navy")
    axs[i].set_title(title, fontsize=10)
    axs[i].axvline(
        transition_step - TRIM_STEPS, color="red", linewidth=1.2
    )  # Solid transition line

    for start, end in stagnation_intervals:
        start_adj = start - TRIM_STEPS
        end_adj = end - TRIM_STEPS
        if start_adj < 0 or end_adj < 0:
            continue
        axs[i].axvspan(start_adj, end_adj, color="gray", alpha=0.15)
        axs[i].axvline(start_adj, color="black", linestyle="--")
        axs[i].axvline(end_adj, color="blue", linestyle="--")

    axs[i].set_ylabel("Value")
    axs[i].grid(True)

axs[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.show()
