# stagnation_classifier/plot_episode_data.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class EpisodePlotter:
    def __init__(self, episode_data):
        self.raw = episode_data["data"]
        self.transition_step = episode_data.get("transition_step", -1)
        self._extract_signals()

    def _extract_signals(self):
        self.vel_x = [step["nonpredictable"][7, -1] for step in self.raw]
        self.vel_z = [step["nonpredictable"][8, -1] for step in self.raw]
        self.pos_x = [step["nonpredictable"][9, -1] for step in self.raw]
        self.pos_z = [step["nonpredictable"][10, -1] for step in self.raw]
        self.th_y = [step["predictable"][4, -1] for step in self.raw]
        self.dth_y = [step["predictable"][5, -1] for step in self.raw]
        self.steps = list(range(len(self.raw)))

    def plot(self):
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(self.steps, self.vel_x, label="base_vel_x")
        plt.plot(self.steps, self.vel_z, label="base_vel_z")
        plt.title("Base Velocities")
        plt.axvline(self.transition_step, color="r", linestyle="--", label="transition")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(self.steps, self.pos_x, label="base_pos_x")
        plt.plot(self.steps, self.pos_z, label="base_pos_z")
        plt.title("Base Positions")
        plt.axvline(self.transition_step, color="r", linestyle="--")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(self.steps, self.th_y, label="base_theta_y")
        plt.plot(self.steps, self.dth_y, label="base_dtheta_y")
        plt.title("Base Orientation (Yaw) and Angular Velocity")
        plt.axvline(self.transition_step, color="r", linestyle="--")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage if run as script
if __name__ == "__main__":
    episode_file = "stagnation_classifier/raw_episodes/episode_000.pkl"
    with open(episode_file, "rb") as f:
        episode_data = pickle.load(f)

    plotter = EpisodePlotter(episode_data)
    plotter.plot()
