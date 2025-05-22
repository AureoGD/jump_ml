import os
import pickle
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from jump_modular_env.jump_env import JumperEnv
from stagnation_classifier.cnn_stagnation_wrapper import StagnationClassifier

# --- Configuration ---
MAX_STEPS = 3000
TRIM_STEPS = 25
MODEL_PATH = "stagnation_classifier/models/cnn_classifier_IV.pth"

# --- Initialize ---
env = JumperEnv(render=True, disturb=False, policy_type="multihead")
classifier = StagnationClassifier(MODEL_PATH)

# --- Helper ---
ml = env.get_wrapper_attr("ml_wrapper")

# --- Transition logic ---
transition_step = 1500
mode_1 = 0
mode_2 = 2

obs, _ = env.reset()
episode_log = []
stagnation_preds = []
stagnation_intervals = []

# --- Start video ---
video_path = "stagnation_classifier/tmp_episode.mp4"
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)

is_stagnated = False
stagnation_start = None

for step in range(MAX_STEPS):
    action = [mode_1 if step < transition_step else mode_2]
    obs, reward, terminated, truncated, info = env.step(action)

    window = {
        "base_past": np.stack(list(ml.base_past), axis=1),
        "joint_past": np.stack(list(ml.joint_past), axis=1),
        "comp_past": np.stack(list(ml.comp_states), axis=1),
        "base_future": ml.pred_st[0:10, :],
        "joint_future": ml.pred_st[10:, :],
    }
    episode_log.append(window)

    pred = classifier.predict(ml.base_past, ml.joint_past, ml.comp_states)
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

# --- Plot ---
timesteps = np.arange(len(episode_log[TRIM_STEPS:]))
pos_x = [step["comp_past"][2, -1] for step in episode_log[TRIM_STEPS:]]
vel_x = [step["comp_past"][0, -1] for step in episode_log[TRIM_STEPS:]]
theta_y = [step["base_past"][4, -1] for step in episode_log[TRIM_STEPS:]]
dtheta_y = [step["base_past"][5, -1] for step in episode_log[TRIM_STEPS:]]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axs = axs.flatten()

signals = [
    (pos_x, "Base Pos X"),
    (vel_x, "Base Vel X"),
    (theta_y, "Theta Y"),
    (dtheta_y, "DTheta Y"),
]

for i, (signal, title) in enumerate(signals):
    axs[i].plot(timesteps, signal, color="black")
    axs[i].set_title(title)

    axs[i].axvline(transition_step - TRIM_STEPS, color="red", linestyle="--")

    for start, end in stagnation_intervals:
        start_adj = start - TRIM_STEPS
        end_adj = end - TRIM_STEPS
        if start_adj < 0 or end_adj < 0:
            continue
        axs[i].axvspan(start_adj, end_adj, color="gray", alpha=0.2)

    axs[i].grid(True)

plt.tight_layout()
plt.show()
