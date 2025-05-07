import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# -------------------
# CONFIG
# -------------------
file_path = "stagnation_classifier/raw_episodes/episode_039.pkl"
save_dir = "stagnation_classifier/labeled_episodes_manual"
os.makedirs(save_dir, exist_ok=True)

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
# EXTRACT SIGNALS
# -------------------
signals = {
    "pos_x": np.array([step["nonpredictable"][9, -1, 0] for step in raw_trimmed]),
    "pos_z": np.array([step["nonpredictable"][10, -1, 0] for step in raw_trimmed]),
    "theta_y": np.array([step["predictable"][4, -1, 0] for step in raw_trimmed]),
    "vel_x": np.array([step["nonpredictable"][7, -1, 0] for step in raw_trimmed]),
    "vel_z": np.array([step["nonpredictable"][8, -1, 0] for step in raw_trimmed]),
    "dtheta_y": np.array([step["predictable"][5, -1, 0] for step in raw_trimmed]),
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


# Run selection for each signal
for name, sig in signals.items():
    print(f"\n🖱️  Select stagnation for {name}. Close the window to continue.")
    select_stagnation(name, sig)

# -------------------
# COMBINE SELECTIONS
# -------------------
masks = []
for name in signals:
    mask = np.zeros(T, dtype=bool)
    for start, end in selections[name]:
        mask[start:end] = True
    masks.append(mask)

final_mask = np.logical_and.reduce(masks).astype(int)

# -------------------
# FINAL VISUALIZATION
# -------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, signals["pos_x"], label="pos_x", color="black")
ax.axvline(transition_step - 50, color="red", linestyle="--", label="Transition")
ax.set_title("Final Combined Stagnation Regions (Intersection)")
ax.set_xlabel("Time Step")
ax.set_ylabel("pos_x")
ax.grid(True)

# Shade regions
in_stag = False
for t in range(T):
    if final_mask[t] and not in_stag:
        start = t
        in_stag = True
    elif not final_mask[t] and in_stag:
        ax.axvspan(start, t, color="gray", alpha=0.3)
        in_stag = False
if in_stag:
    ax.axvspan(start, T, color="gray", alpha=0.3)

ax.legend()
plt.tight_layout()
plt.show()

# -------------------
# SAVE LABELED EPISODE
# -------------------
labeled_data = []
for i in range(T):
    labeled_data.append(
        {
            "predictable": raw_trimmed[i]["predictable"],
            "nonpredictable": raw_trimmed[i]["nonpredictable"],
            "label": int(final_mask[i]),
        }
    )

base_name = os.path.basename(file_path).replace("episode", "use_label")
save_path = os.path.join(save_dir, base_name)

with open(save_path, "wb") as f:
    pickle.dump(labeled_data, f)

print(f"\n✅ Labeled data saved to: {save_path}")
