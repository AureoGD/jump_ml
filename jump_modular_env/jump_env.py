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
from .ml_wrapper_v2 import MLWrapper


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
        self.discreate_actions = discrete_actions

        self.robot_states = RobotStates()
        self.robot_model = JumpModel(_robot_states=self.robot_states)
        self.ml_wrapper = MLWrapper(_robot_states=self.robot_states, debug=self.debug, policy_type=policy_type)
        self.robot_model.set_pred_states_dim(self.ml_wrapper.NUM_OBS_STATES_P, self.ml_wrapper.NF, augmented=True)

        self.physics = PhysicsEngine(self.robot_model, render)
        self.disturbance = Disturbance(self.robot_model)
        self.logger = Logger(log_interval)

        self.interations = int(self.robot_model.rgc_dt / self.robot_model.sim_dt)

        if self.discreate_actions:
            self.action_space = gym.spaces.Discrete(self.ml_wrapper.NUM_MODES)
        else:
            self.action_space = gym.spaces.Box(
                low=0.0,
                high=float(self.ml_wrapper.NUM_MODES - 1),
                shape=(1, ),
                dtype=np.float32,
            )

        self._setup_observation_space()
        self.episode_reward = 0
        self.current_step = 0
        self.ep = 0

    def _setup_observation_space(self):
        if self.policy_type == "cnn":
            self.observation_space = spaces.Dict({
                "predictable":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        self.ml_wrapper.NUM_OBS_STATES_P,
                        self.ml_wrapper.NP + self.ml_wrapper.NF,
                    ),
                    dtype=np.float32,
                ),
                "nonpredictable":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        self.ml_wrapper.NUM_OBS_STATES_NP,
                        self.ml_wrapper.NP,
                    ),
                    dtype=np.float32,
                ),
            })
        elif self.policy_type == "cnn3h":
            self.observation_space = spaces.Dict({
                "pred_past":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        self.ml_wrapper.NUM_OBS_STATES_P,
                        self.ml_wrapper.NP,
                    ),
                    dtype=np.float32,
                ),
                "pred_fut":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        self.ml_wrapper.NUM_OBS_STATES_P,
                        self.ml_wrapper.NF,
                    ),
                    dtype=np.float32,
                ),
                "nonpredictable":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        self.ml_wrapper.NUM_OBS_STATES_NP,
                        self.ml_wrapper.NP,
                    ),
                    dtype=np.float32,
                ),
            })
        elif self.policy_type == "multihead":
            self.observation_space = spaces.Dict({
                "base_past":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        10,
                        self.ml_wrapper.NP,
                    ),  # hard code for now [r, dr, b, db, th, dth]
                    dtype=np.float32,
                ),
                "base_future":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        10,
                        self.ml_wrapper.NF,
                    ),  # hard code for now [r, dr, b, db, th, dth]
                    dtype=np.float32,
                ),
                "joint_past":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        12,
                        self.ml_wrapper.NP,
                    ),  # hard code for now [q, dq, qr, tau]
                    dtype=np.float32,
                ),
                "joint_future":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        12,
                        self.ml_wrapper.NF,
                    ),  # hard code for now [q, dq, qr, tau]
                    dtype=np.float32,
                ),
                "comp_past":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(
                        5,
                        self.ml_wrapper.NP,
                    ),  # hard code for now [stag, phase, trans_h, foot_st, sucess]
                    dtype=np.float32,
                ),
            })
        else:
            # Flattened MLP input
            self.observation_space = spaces.Dict({
                "predictable":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(self.ml_wrapper.NUM_OBS_STATES_NP * (self.ml_wrapper.NP + 1 + self.ml_wrapper.NF), ),
                    dtype=np.float32,
                ),
                "nonpredictable":
                spaces.Box(
                    low=self.ml_wrapper.OBS_LOW_VALUE,
                    high=self.ml_wrapper.OBS_HIGH_VALUE,
                    shape=(self.ml_wrapper.NUM_OBS_STATES_NP * (self.ml_wrapper.NP + 1), ),
                    dtype=np.float32,
                ),
            })

    def step(self, action):
        # mode = self.robot_model.state_machine(action)
        # self.robot_model.new_action(mode)

        if isinstance(self.action_space, gym.spaces.Discrete):
            # PPO-style: action is an integer index
            action = int(np.clip(action, 0, self.ml_wrapper.NUM_MODES - 1))
        else:
            # SAC-style: action is a float, needs rounding and clipping
            if isinstance(action, (np.ndarray, list)):
                action_value = action[0]
            else:
                action_value = action
            action = int(np.clip(np.round(action_value), 0, self.ml_wrapper.NUM_MODES - 1))

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

        # info = {"Phase": self.ml_wrapper.reward_fcns.curriculum_phase}

        info = {
            "stagnation": self.ml_wrapper.stag_value,
            "success": self.ml_wrapper.success_rate,
            "phase": self.ml_wrapper.reward_fcns.curriculum_phase,
        }

        if terminated or terminated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step,
                "tc": self.ml_wrapper.total_modes_changes,
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

        dummy_action = self.action_space.sample()
        # dummy_action = 0
        for _ in range(self.ml_wrapper.NP):
            obs, _, terminated, truncated, _ = self.step(dummy_action)
            if terminated or truncated:
                break

        self.current_step = 0
        self.episode_reward = 0
        self.ml_wrapper.reset_vars()

        return obs, {"ep": self.ep}
