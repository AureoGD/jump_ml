import numpy as np
import pybullet as p
from collections import deque
from typing import Dict, Any, Deque
from .jump_states import _robot_states
from .hopper_model import HopperModel
from .physics import PhysicsWorld


class Individual():

    def __init__(self, individual_id: int, config: Dict[str, Any]):
        self.individual_id = individual_id
        self.config = config

        self.robot_states = _robot_states()

        self.physics = PhysicsWorld(self.config.get('physics_config', {}))
        self.robot = HopperModel(self.robot_states)

        self.sim_per_rgc_step = int(self.robot.rgc_dt / self.robot.sim_dt)
        self.step_states: Deque[Dict] = deque(maxlen=self.sim_per_rgc_step)
        self.current_episode_step = 0

    def one_step(self, action):
        self.robot.new_action(action=action)
        for _ in range(self.sim_per_rgc_step):
            self.physics.step(self.robot, self.current_episode_step)
            self.step_states.append(self.robot.robot_states.copy())
            self.current_episode_step += 1
        return self.step_states

    def reset(self):
        """Resets the simulation environment and robot state for a new episode."""
        self.current_episode_step = 0
        self.step_states.clear()
        self.robot.reset_variables()
        self.physics.reset_robot(self.robot)

    def close(self):
        """Disconnects from the PyBullet server."""
        self.physics.disconnect()
