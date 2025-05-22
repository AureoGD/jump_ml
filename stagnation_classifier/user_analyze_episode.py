import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# -------------------
# CONFIG
# -------------------
file_path = "stagnation_classifier/raw_episodes/episode_019.pkl"
save_dir = "stagnation_classifier/labeled_episodes_manual"
curve_dir = "stagnation_classifier/labeled_curves"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(curve_dir, exist_ok=True)

# -------------------
# LOAD
# -------------------
with open(file_path, "rb") as f:
    data = pickle.load(f)

raw = data["data"]
transition_step = data["transition_step"]
raw_trimmed = raw[50:]
T = len(raw_trimmed)
time = np.arange(T)

# -------------------
# EXTRACT SIGNALS (adjust indices if needed)
# -------------------
signals = {
    "pos_x": np.array([step["base_past"][0, -1] for step in raw_trimmed]),  # r_x
    "pos_z": np.array([step["base_past"][5, -1] for step in raw_trimmed]),  # b_z
    "theta_y": np.array([step["base_past"][4, -1] for step in raw_trimmed]),  # theta y
    "vel_x": np.array([step["base_past"][1, -1] for step in raw_trimmed]),  # r_vx
    "vel_z": np.array([step["base_past"][3, -1] for step in raw_trimmed]),  # r_vz
    "dtheta_y": np.array([step["base_past"][3, -1] for step in raw_trimmed]),  # dtheta y
}

# -------------------
# INTERACTIVE SELECTION
# -------------------
selections = {key: [] for key in signals.keys()}


def select_stagnation(signal_name, signal_data):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal_data, label=signal_name)
    ax.axvline(transition_step - 50, color="red", linestyle="--", label="Transition")
    ax.set_title(f"Select stagnation regions for {signal_name}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(signal_name)
    ax.grid(True)
    ax.legend()

    def onselect(xmin, xmax):
        i0, i1 = int(xmin), int(xmax)
        selections[signal_name].append((i0, i1))
        ax.axvspan(i0, i1, color="gray", alpha=0.3)
        fig.canvas.draw()
        print(f"[{signal_name}] Region: {i0} to {i1}")

    span = SpanSelector(
        ax,
        onselect,
        "horizontal",
        useblit=True,
        interactive=True,
        props=dict(alpha=0.3, facecolor="gray"),
    )

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# Run interactive selection for each signal
for name, sig in signals.items():
    print(f"\n🖱️  Select stagnation for {name}. Close the window to continue.")
    select_stagnation(name, sig)

# -------------------
# COMBINE SELECTIONS INTO STAGNATION CURVE
# -------------------
n_signals = len(signals)
masks = []

for name in signals:
    mask = np.zeros(T, dtype=float)
    for start, end in selections[name]:
        mask[start:end] = 1.0
    masks.append(mask / n_signals)  # Each signal contributes equally

# Compute stagnation curve
stagnation_curve = np.sum(masks, axis=0)  # Shape (T,), values in [0, 1]

# Save the curve separately (optional)
curve_filename = os.path.basename(file_path).replace("episode", "curve").replace(".pkl", ".npy")
np.save(os.path.join(curve_dir, curve_filename), stagnation_curve)
print(f"📈 Saved stagnation curve to {curve_filename}")

# -------------------
# VISUALIZATION
# -------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, signals["pos_x"], label="pos_x", color="black")
ax.axvline(transition_step - 50, color="red", linestyle="--", label="Transition")
ax.set_title("Stagnation Score Curve (Soft Labels)")
ax.set_xlabel("Time Step")
ax.set_ylabel("pos_x")
ax.grid(True)

# Overlay stagnation curve
ax2 = ax.twinx()
ax2.plot(time, stagnation_curve, label="Stagnation Curve", color="purple", linewidth=2, alpha=0.7)
ax2.set_ylabel("Stagnation Score (0 to 1)")
ax2.set_ylim(0, 1)

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.show()

# -------------------
# SAVE LABELED EPISODE
# -------------------
labeled_data = []
for i in range(T):
    labeled_data.append({
        "base_past": raw_trimmed[i]["base_past"],
        "joint_past": raw_trimmed[i]["joint_past"],
        "comp_past": raw_trimmed[i]["comp_past"],
        "base_future": raw_trimmed[i]["base_future"],
        "joint_future": raw_trimmed[i]["joint_future"],
        "label": float(stagnation_curve[i]),  # Save the soft label directly
    })

base_name = os.path.basename(file_path).replace("episode", "use_label")
save_path = os.path.join(save_dir, base_name)

with open(save_path, "wb") as f:
    pickle.dump(labeled_data, f)

print(f"\n✅ Labeled data saved to: {save_path}")
