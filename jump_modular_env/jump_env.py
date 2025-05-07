# jump_modular_env/
# |- jump_env.py       <- Main Gym Environment class
# |- physics_engine.py   <- Handles PyBullet setup and dynamics
# |- disturbance.py      <- Manages random external forces
# |- logger.py           <- Saves robot and training data

# --- jump_env.py ---
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .physics_engine import PhysicsEngine
from .disturbance import Disturbance
from .logger import Logger
from .jump_model import RobotStates, JumpModel
from .ml_wrapper import MLWrapper


class JumperEnv(gym.Env):
    def __init__(
        self,
        policy_type="mlp",
        render=False,
        render_every=False,
        render_interval=10,
        log_interval=10,
        debug=False,
        disturb=False,
        discrete_actions=True,
    ):
        super().__init__()

        self.policy_type = policy_type.lower()
        self.debug = debug
        self.render_every = render_every
        self.render_interval = render_interval
        self.disturb_enabled = disturb

        self.robot_states = RobotStates()
        self.robot_model = JumpModel(_robot_states=self.robot_states)
        self.ml_wrapper = MLWrapper(_robot_states=self.robot_states, debug=self.debug)
        self.robot_model.set_pred_states_dim(
            self.ml_wrapper.NUM_OBS_STATES_P, self.ml_wrapper.NF
        )

        self.physics = PhysicsEngine(self.robot_model, render)
        self.disturbance = Disturbance(self.robot_model)
        self.logger = Logger(log_interval)

        self.interations = int(self.robot_model.rgc_dt / self.robot_model.sim_dt)

        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._setup_observation_space()
        self.episode_reward = 0
        self.current_step = 0
        self.ep = 0

    def _setup_observation_space(self):
        if self.policy_type == "cnn":
            self.observation_space = spaces.Dict(
                {
                    "predictable": spaces.Box(
                        low=self.ml_wrapper.OBS_LOW_VALUE,
                        high=self.ml_wrapper.OBS_HIGH_VALUE,
                        shape=(
                            self.ml_wrapper.NUM_OBS_STATES_P,
                            self.ml_wrapper.NP + self.ml_wrapper.NF,
                        ),
                        dtype=np.float32,
                    ),
                    "nonpredictable": spaces.Box(
                        low=self.ml_wrapper.OBS_LOW_VALUE,
                        high=self.ml_wrapper.OBS_HIGH_VALUE,
                        shape=(
                            self.ml_wrapper.NUM_OBS_STATES_NP,
                            self.ml_wrapper.NP,
                        ),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            # Flattened MLP input
            self.observation_space = spaces.Dict(
                {
                    "predictable": spaces.Box(
                        low=self.ml_wrapper.OBS_LOW_VALUE,
                        high=self.ml_wrapper.OBS_HIGH_VALUE,
                        shape=(
                            self.ml_wrapper.NUM_OBS_STATES_NP
                            * (self.ml_wrapper.NP + 1 + self.ml_wrapper.NF),
                        ),
                        dtype=np.float32,
                    ),
                    "nonpredictable": spaces.Box(
                        low=self.ml_wrapper.OBS_LOW_VALUE,
                        high=self.ml_wrapper.OBS_HIGH_VALUE,
                        shape=(
                            self.ml_wrapper.NUM_OBS_STATES_NP
                            * (self.ml_wrapper.NP + 1),
                        ),
                        dtype=np.float32,
                    ),
                }
            )

    def step(self, action):
        # mode = self.robot_model.state_machine(action)
        # self.robot_model.new_action(mode)

        if isinstance(self.action_space, gym.spaces.Discrete):
            action = int(np.clip(action, 0, self.ml_wrapper.NUM_MODES - 1))
        else:
            # Action might be float or array, be safe:
            if isinstance(action, (np.ndarray, list)):
                action_value = action[0]
            else:
                action_value = action
            action = int(
                np.clip(np.round(action_value), 0, self.ml_wrapper.NUM_MODES - 1)
            )

        self.robot_model.new_action(action)

        for _ in range(self.interations):
            q, dq, f = self.physics.get_joint_state()
            self.robot_model.update_robot_states(q, dq, f)

            tau = self.robot_model.command_torque()
            self.physics.apply_torque(tau)

            if self.disturb_enabled:
                self.disturbance.apply(self.current_step)

            self.physics.step()

        q, dq, f = self.physics.get_joint_state()
        self.robot_model.update_robot_states(q, dq, f)
        self.robot_model.ml_states()
        self.ml_wrapper.update_pred_states(self.robot_model.pred_states)

        obs, reward, terminated = self.ml_wrapper.end_of_step()

        self.current_step += 1
        self.episode_reward += reward

        info = {}
        if terminated or terminated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step,
            }

        return obs, float(np.squeeze(reward)), terminated, terminated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ep += 1

        self.robot_model.reset_variables()
        self.physics.reset()

        q = self.robot_model.randon_joint_pos()
        self.physics.initialize_joint_states(q)

        if self.disturb_enabled:
            self.disturbance.reset()
            self.disturbance.update_model(self.physics.model)

        self.robot_model.init_qr(q[3:])
        q, dq, f = self.physics.get_joint_state()
        self.robot_model.update_robot_states(q, dq, f)
        self.robot_model.ml_states()

        dummy_action = 0  # or self.action_space.sample()
        for _ in range(self.ml_wrapper.NP):
            obs, _, terminated, truncated, _ = self.step(dummy_action)
            if terminated or truncated:
                break

        self.current_step = 0
        self.episode_reward = 0
        self.ml_wrapper.reset_vars()

        return obs, {"ep": self.ep}
