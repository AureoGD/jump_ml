import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class StagnationAnalyzer:
    def __init__(
        self,
        file_path,
        trim_steps=75,
        vel_thresh=0.005,
        dth_thresh=0.005,
        pos_thresh=0.0015,
        ori_thresh=0.005,
        min_steps=150,
        save_dir="stagnation_classifier/labeled_episodes",
    ):
        self.file_path = file_path
        self.trim_steps = trim_steps
        self.vel_thresh = vel_thresh
        self.dth_thresh = dth_thresh
        self.pos_thresh = pos_thresh
        self.ori_thresh = ori_thresh
        self.min_steps = min_steps
        self.save_dir = save_dir

        self.data_dict = None
        self.raw = None
        self.timesteps = None
        self.stagnation_windows = []

        os.makedirs(self.save_dir, exist_ok=True)

    def load(self):
        with open(self.file_path, "rb") as f:
            self.data_dict = pickle.load(f)
        self.raw = self.data_dict["data"][self.trim_steps :]
        self.timesteps = np.arange(len(self.raw))

    def extract_signals(self):
        self.vel_x = np.array([step["nonpredictable"][7, -1] for step in self.raw])
        self.vel_z = np.array([step["nonpredictable"][8, -1] for step in self.raw])
        self.pos_x = np.array([step["nonpredictable"][9, -1] for step in self.raw])
        self.pos_z = np.array([step["nonpredictable"][10, -1] for step in self.raw])
        self.th_y = np.array([step["predictable"][4, -1] for step in self.raw])
        self.dth_y = np.array([step["predictable"][5, -1] for step in self.raw])

        ts = self.data_dict.get("transition_step", None)
        m1 = self.data_dict.get("mode_1", 0.0)
        m2 = self.data_dict.get("mode_2", 2.0)
        self.mode_signal = [
            m1 if i < (ts - self.trim_steps if ts else len(self.raw)) else m2
            for i in range(len(self.raw))
        ]

    def detect_stagnation(self):
        vel_flags = (
            (np.abs(self.vel_x) < self.vel_thresh)
            & (np.abs(self.vel_z) < self.vel_thresh)
            & (np.abs(self.dth_y) < self.dth_thresh)
        )

        pos_x_diff = np.abs(np.diff(np.insert(self.pos_x, 0, self.pos_x[0])))
        pos_z_diff = np.abs(np.diff(np.insert(self.pos_z, 0, self.pos_z[0])))
        ori_var = np.abs(np.diff(np.insert(self.th_y, 0, self.th_y[0])))
        pos_var = pos_x_diff + pos_z_diff
        posori_flags = (pos_var < self.pos_thresh) & (ori_var < self.ori_thresh)

        flags = posori_flags

        self.stagnation_windows = []
        start = None
        current_mode = self.mode_signal[0]

        for i in range(len(flags)):
            if flags.all() and self.mode_signal[i] == current_mode:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= self.min_steps:
                    self.stagnation_windows.append((start, i))
                start = None
                current_mode = self.mode_signal[i]

        if start is not None and len(flags) - start >= self.min_steps:
            self.stagnation_windows.append((start, len(flags)))

    def generate_labels(self):
        labels = np.zeros(len(self.raw), dtype=int)
        for start, end in self.stagnation_windows:
            labels[start:end] = 1
        return labels

    def save_labeled_episode(self):
        labels = self.generate_labels()
        labeled_data = [
            {
                "predictable": step["predictable"],
                "nonpredictable": step["nonpredictable"],
                "label": labels[i],
            }
            for i, step in enumerate(self.raw)
        ]

        base_name = os.path.basename(self.file_path).replace("episode", "labeled")
        save_path = os.path.join(self.save_dir, base_name)

        with open(save_path, "wb") as f:
            pickle.dump(labeled_data, f)

        print(f"✅ Labeled episode saved to: {save_path}")

    def plot(self):
        fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        axs = axs.flatten()

        signals = [
            (self.pos_x, "Base Position X"),
            (self.vel_x, "Base Velocity X"),
            (self.pos_z, "Base Position Z"),
            (self.vel_z, "Base Velocity Z"),
            (self.th_y, "Base Orientation (Yaw)"),
            (self.dth_y, "Base Angular Velocity (Yaw Rate)"),
        ]

        ts = self.data_dict.get("transition_step", None)
        ts_adj = ts - self.trim_steps if ts else None

        for i, (signal, label) in enumerate(signals):
            axs[i].plot(self.timesteps, signal, label=label)
            axs[i].set_ylabel(label)
            axs[i].grid(True)

            if ts_adj and 0 <= ts_adj < len(self.timesteps):
                axs[i].axvline(ts_adj, color="r", linestyle="--", label="Transition")

            for start, end in self.stagnation_windows:
                axs[i].axvspan(start, end, color="gray", alpha=0.2)
                axs[i].axvline(start, color="blue", linestyle="--")
                axs[i].axvline(end, color="purple", linestyle="--")

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout()
        plt.show()

    def run(self):
        self.load()
        self.extract_signals()
        self.detect_stagnation()
        self.plot()
        self.save_labeled_episode()


# Example usage:
analyzer = StagnationAnalyzer("stagnation_classifier/raw_episodes/episode_000.pkl")
analyzer.run()
