import numpy as np
import os
from math import cos, sin, sqrt
import random
from collections import deque
import sys
import time

# Assuming jump_matrices.py contains the ModelMatrices class
from .jump_matrices import ModelMatrices

# --- Path setup for RGC controller ---
# This is a more robust way to define the path to your compiled C++ module
# The `ls` command confirms the build directory is at '~/jump_ml/rgc_controller/build'.
# We use an absolute path to be safe. os.path.expanduser('~') resolves '~' to your home directory.
# TODO: solve the path to be generic
path_to_build = os.path.expanduser('~/jump_ml/rgc_controller/build')
path_to_config = os.path.expanduser('~/jump_ml/rgc_controller/config/config.yaml')

# Add the correct path to where Python searches for modules
if path_to_build not in sys.path:
    sys.path.insert(0, path_to_build)

# Now, Python will look in the correct directory.
try:
    import pybind_opWrapper
    print("Successfully imported pybind_opWrapper!")
except ImportError:
    print(f"FATAL ERROR: Could not import 'pybind_opWrapper'.")
    print(f"Python looked in the path: {path_to_build}")
    print("Please double-check that this path is correct and the .so file exists there.")
    sys.exit(1)


class HopperModel():

    def __init__(self, robot_states_dict: dict):
        """
        Args:
            robot_states_dict (dict): A dictionary to hold and share the robot's state.
        """
        # --- Constants and Model Setup ---
        self.AC_JOINT_LIST = (4, 6, 8)
        self.JOINT_ST_LIST = (0, 1, 2, 4, 6, 8)
        self.CONTACT_JOINTS = (10, 11)
        self.JOINT_MODEL_NUM = len(self.JOINT_ST_LIST)
        self.mdl = ModelMatrices()
        self.sim_dt = 0.001
        self.rgc_dt = 0.01

        # --- PD Controller Gains ---
        self.kp = 150
        self.kd = 10
        self.Kp = self.kp * np.identity(len(self.AC_JOINT_LIST))
        self.Kd = self.kd * np.identity(len(self.AC_JOINT_LIST))
        self.MAX_TAU = 50
        self.V_MAX_TAU = self.MAX_TAU * np.ones((len(self.AC_JOINT_LIST), 1))

        # --- RGC Controller Setup ---
        self.RGC = pybind_opWrapper.Op_Wrapper()
        self.RGC.load_config(path_to_config)
        self.RGC.RGCConfig(self.rgc_dt, self.kp, self.kd)

        # --- State Management ---
        self.robot_states = robot_states_dict  # This is now a dictionary

        # Ensure keys used in this class exist in the state dictionary
        self.robot_states.setdefault('dqr', np.zeros((3, 1), dtype=np.float32))
        self.robot_states.setdefault('cont_j', np.zeros((1, 1), dtype=np.float32))
        self.robot_states.setdefault('cont_st', np.zeros((1, 1), dtype=np.float32))

        self.rw_ant = np.zeros((2, 1), dtype=np.float64)  # For internal velocity calculation
        self.flag_first_int = False
        self.ac_step = 0

    def new_action(self, action: int):
        """Applies a new MPC mode action to the RGC controller."""
        self.robot_states['cont_mode'][0, 0] = action  # Use 'cont_mode' as defined in your jump_states

        self.RGC.UpdateSt(
            self.robot_states['q'],
            self.robot_states['dq'],
            self.robot_states['qr'],
            self.robot_states['r_vel'],
            self.robot_states['r_pos'],
            self.robot_states['b_vel'],
            self.robot_states['b_pos'],
            self.robot_states['dth'][0, 0],
            self.robot_states['th'][0, 0],
        )

        rgc_return = self.RGC.ChooseRGCPO(action)

        if rgc_return == 1:
            delta_qr = self.RGC.delta_qr.reshape(3, 1)
            self.robot_states['cont_st'][0, 0] = 1
            if np.any(np.isnan(delta_qr)):
                self.robot_states['cont_st'][0, 0] = 0
            else:
                self.robot_states['qr'][:, 0] = self.robot_states['qr'][:, 0] + delta_qr[:, 0]
                self.robot_states['cont_j'][0, 0] = self.RGC.obj_val
        else:
            self.robot_states['cont_st'][0, 0] = -1

    def command_torque(self) -> np.ndarray:
        """Computes the command torque from PD and gravity compensation."""
        tau_pd = self._torque_pd()
        tau_cg = self._torque_compG()
        self.robot_states['tau'] = np.clip(tau_pd - tau_cg, -self.V_MAX_TAU, self.V_MAX_TAU)
        return self.robot_states['tau']

    def _torque_pd(self) -> np.ndarray:
        """Calculates the Proportional-Derivative control torque."""
        # Note: 'dqr' (desired joint velocity) is assumed to be zero unless set otherwise.
        # This was part of your original RobotStates but not in the new jump_states.py
        return self.Kp @ (self.robot_states['qr'] - self.robot_states['q']) + \
               self.Kd @ (self.robot_states['dqr'] - self.robot_states['dq'])

    def _torque_compG(self) -> np.ndarray:
        """Calculates gravity compensation torque."""
        rot = self._rotY()

        J_com1 = rot @ self.mdl.J_com1
        J_com2 = rot @ self.mdl.J_com2
        J_com3 = rot @ self.mdl.J_com3
        tau_g = (J_com1.transpose() * self.mdl.m1 + J_com2.transpose() * self.mdl.m2 +
                 J_com3.transpose() * self.mdl.m3) @ np.array([[0], [0], [-9.81]])
        return tau_g

    def update_robot_states(self, q_aux: np.ndarray, dq_aux: np.ndarray, aux_F_cont: np.ndarray):
        """Updates the state dictionary from raw simulation data."""
        self.robot_states['b_pos'][0, 0] = q_aux[0, 0]
        self.robot_states['b_pos'][1, 0] = q_aux[1, 0]
        self.robot_states['b_vel'][0, 0] = dq_aux[0, 0]
        self.robot_states['b_vel'][1, 0] = dq_aux[1, 0]
        self.robot_states['th'][0, 0] = q_aux[2, 0]
        self.robot_states['dth'][0, 0] = dq_aux[2, 0]

        self.robot_states['q'][:, 0] = q_aux[3:6, 0]
        self.robot_states['dq'][:, 0] = dq_aux[3:6, 0]

        self.robot_states['heel_cont'][0, 0] = aux_F_cont[1, 0]
        self.robot_states['toe_cont'][0, 0] = aux_F_cont[0, 0]
        self.robot_states['foot_st'][0, 0] = (aux_F_cont[0, 0] + aux_F_cont[0, 0] * 2) / 3

        # Update the underlying matrix model with new joint states
        self.mdl.update_robot_states(q=self.robot_states['q'], dq=self.robot_states['dq'])
        self.mdl.update_kinematics()

        # Update kinematic properties and CoM
        r_vet = self.mdl.update_com_pos()
        self.robot_states['r_pos'], self.robot_states['r_vel'] = self._write_wrt(r_vet, self.robot_states['r_pos'])
        if not self.flag_first_int:
            self.flag_first_int = True
            self.robot_states['r_vel'][:] = 0

    def _rotY(self) -> np.ndarray:
        """Creates a rotation matrix around the Y-axis."""
        th = self.robot_states['th'][0, 0]
        return np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])

    def randon_joint_pos(self) -> np.ndarray:
        """Generates a random initial joint configuration for an episode."""
        q = np.zeros((self.JOINT_MODEL_NUM, 1), dtype=np.float64)
        q[0, 0] = random.uniform(0, 2)
        q[1, 0] = 0.88
        q[3, 0] = -0.56
        q[4, 0] = 1.06
        q[5, 0] = -0.50

        self.rw_ant[:] = 0
        self.flag_first_int = False
        return q

    def _write_wrt(self, pos_local: np.ndarray, pos_global_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transforms a local position to global and calculates velocity."""
        pos_global = np.zeros((2, 1), dtype=np.float64)
        pos_rotated = self._rotY() @ pos_local.reshape(3, 1)

        pos_global[0, 0] = self.robot_states['b_pos'][0, 0] + pos_rotated[0, 0]
        pos_global[1, 0] = self.robot_states['b_pos'][1, 0] + pos_rotated[2, 0]

        vel_global = (pos_global - pos_global_prev) / self.sim_dt
        return pos_global, vel_global

    def reset_variables(self):
        """Resets variables at the start of a new episode."""
        self.ac_step = 0
        self.RGC.ResetPO()
        self.flag_first_int = False

    def init_qr(self, q_initial_joints: np.ndarray):
        """Initializes the reference joint positions (qr) based on the initial configuration."""
        self.robot_states['qr'][:, 0] = q_initial_joints[:, 0] - np.random.uniform(-0.1, 0.1, 3)
