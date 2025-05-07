import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# Load data
file_path = "stagnation_classifier/raw_episodes/episode_000.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

raw = data["data"]
transition_step = data["transition_step"]

# Extract and remove first 50 steps
signals = {
    "pos_x": np.array([step["nonpredictable"][9, -1, 0] for step in raw])[50:],
    "pos_z": np.array([step["nonpredictable"][10, -1, 0] for step in raw])[50:],
    "theta_y": np.array([step["predictable"][4, -1, 0] for step in raw])[50:],
    "vel_x": np.array([step["nonpredictable"][7, -1, 0] for step in raw])[50:],
    "vel_z": np.array([step["nonpredictable"][8, -1, 0] for step in raw])[50:],
    "dtheta_y": np.array([step["predictable"][5, -1, 0] for step in raw])[50:],
}
T = len(next(iter(signals.values())))
time = np.arange(T)

# Store selections for each signal
selections = {key: [] for key in signals.keys()}


# SpanSelector handler (stateful inside loop)
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
        direction="horizontal",
        useblit=True,
        interactive=True,
        props=dict(alpha=0.3, facecolor="gray"),
    )

    plt.tight_layout()
    plt.show()


# Run selection loop
for name, data_array in signals.items():
    select_stagnation(name, data_array)

# Convert selected ranges into per-step masks
masks = []
for name in signals.keys():
    mask = np.zeros(T, dtype=bool)
    for start, end in selections[name]:
        mask[start:end] = True
    masks.append(mask)

# Final intersection mask
final_mask = np.logical_and.reduce(masks)

# Final visualization
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, signals["pos_x"], label="pos_x", color="black")
ax.set_title("Final Combined Stagnation Regions (Intersection)")
ax.set_xlabel("Time Step")
ax.set_ylabel("pos_x")
ax.grid(True)
ax.axvline(transition_step - 50, color="red", linestyle="--", label="Transition")

# Shade detected stagnation regions
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
