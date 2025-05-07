# jump_modular_env/disturbance.py

import pybullet as p
import numpy as np


class Disturbance:
    def __init__(
        self, model=None, link_index=2, force_magnitude=100.0, duration=10, chance=0.05
    ):
        self.model = model  # Robot model ID (from PyBullet)
        self.link_index = link_index  # Usually base link index
        self.force_magnitude = force_magnitude
        self.duration = duration
        self.chance = chance

        self.remaining = 0
        self.force = None
        self.applied = False

    def update_model(self, new_model):
        """
        Update the internal model reference (e.g., after reset).
        """
        self.model = new_model

    def reset(self):
        """
        Reset internal state each episode.
        """
        self.remaining = 0
        self.force = None
        self.applied = False

    def apply(self, step):
        if self.model is None:
            return  # Model not yet set

        # Start disturbance
        if not self.applied and step > 600:
            if np.random.rand() < self.chance:
                fx = np.random.uniform(-1, 1)
                fz = np.random.uniform(-1, 1)
                self.force = np.array([fx, 0.0, fz])
                self.force /= np.linalg.norm(self.force)
                self.force *= self.force_magnitude
                self.remaining = self.duration
                self.applied = True
                print(f"[Disturbance] Applied force {self.force} at step {step}")

        # Apply if active
        if self.remaining > 0:
            p.applyExternalForce(
                objectUniqueId=self.model,
                linkIndex=self.link_index,
                forceObj=self.force.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
            )
            self.remaining -= 1
