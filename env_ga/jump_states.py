import numpy as np


def _robot_states() -> dict:
    """
    Creates a dictionary to hold all possible state variables for the robot,
    initialized to their default (usually zero) values.
    """
    states = {
        # Base and CoM states
        "b_pos": np.zeros((2, 1), dtype=np.float32),
        "b_vel": np.zeros((2, 1), dtype=np.float32),
        "r_pos": np.zeros((2, 1), dtype=np.float32),
        "r_vel": np.zeros((2, 1), dtype=np.float32),
        "th": np.zeros((1, 1), dtype=np.float32),
        "dth": np.zeros((1, 1), dtype=np.float32),

        # Joint states
        "q": np.zeros((3, 1), dtype=np.float32),
        "dq": np.zeros((3, 1), dtype=np.float32),
        "qr": np.zeros((3, 1), dtype=np.float32),
        "tau": np.zeros((3, 1), dtype=np.float32),

        # Contact states
        "toe_cont": np.zeros((1, 1), dtype=np.float32),
        "heel_cont": np.zeros((1, 1), dtype=np.float32),

        # Toe states
        "toe_pos": np.zeros((2, 1), dtype=np.float64),
        "toe_vel": np.zeros((2, 1), dtype=np.float64),

        # Toe states
        "heel_pos": np.zeros((2, 1), dtype=np.float64),
        "heel_vel": np.zeros((2, 1), dtype=np.float64),

        # Toe states
        "ankle_pos": np.zeros((2, 1), dtype=np.float64),
        "ankle_vel": np.zeros((2, 1), dtype=np.float64),

        # Toe states
        "knee_pos": np.zeros((2, 1), dtype=np.float64),
        "knee_vel": np.zeros((2, 1), dtype=np.float64),

        # foot state
        "foot_st": np.zeros((1, 1), dtype=np.float32),

        # Controller/System states
        "cont_mode": np.zeros((1, 1), dtype=np.float32),
        "cont_j": np.zeros((1, 1), dtype=np.float32),
        "cont_st": np.zeros((1, 1), dtype=np.float32),
    }
    return states
