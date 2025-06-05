# In: env_ga/reward_wrapper.py
import numpy as np
from collections import deque
from typing import Dict, Any

# Using ic from icecream can be helpful for debugging, but if it's not a standard
# dependency, you might want to handle its import gracefully or remove it.
try:
    from icecream import ic
except ImportError:  # graceful fallback if ic isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)


class RewardFcns:

    def __init__(self, robot_states: Dict[str, np.ndarray], robot_model_ref=None, debug=False):
        """
        Calculates various reward components based on the robot's state.

        Args:
            robot_states (Dict): A dictionary containing the robot's current state.
            robot_model_ref (Any, optional): A reference to the robot model, in case
                                             some rewards need to access model parameters. Defaults to None.
            debug (bool, optional): Flag for debug prints. Defaults to False.
        """
        self.robot_states = robot_states
        self.robot_model_ref = robot_model_ref  # Store if needed for complex rewards

        self.rewards = np.zeros((20, 1), dtype=np.float64)

        # states variables
        self.transition_history = 0
        self.foot_contact_state = 0
        self.stagnation_metric = 0
        self.total_modes_changes = 0
        self.n_int = 0
        """ reward weights  """
        self.body_height_weight = 2
        self.body_orientation_weight = 3.5
        self.po_violation_weight = 1.5
        self.stagnation_penalty_weight = 10.0
        self.stand_reward_weight = 2
        self.crouch_weight = 1.0
        # ... other weights ...
        """ rewards auxiliary constants"""
        self.th_threshold = 0.65
        self.max_th = 1.2
        self.prohibited_actions = [6, 7]  # Example modes that are not allowed
        self.phase1_success_steps_threshold = 50
        self.max_com_drop = 0.2
        self.base_threshold = 0.002
        self.multiplier = 1.5
        self.stagnation_threshold = 150
        """ rewards auxiliary variables """
        self.reset_variables()

    def reset_variables(self):
        self.n_int = 0
        self.standing_com_height = 0
        self.curriculum_phase = 1
        self.stagnation_steps = 0
        self.phase1_success_steps = 0
        self.prev_states = None
        self.delta_history = deque(maxlen=30)
        # ... other resets ...

    def reward(self) -> float:
        """Evaluate all rewards terms and sum the values."""
        self.n_int += 1
        self._update_stagnation_metric()

        self.rewards.fill(0)  # Use .fill(0) which is slightly more efficient

        self.rewards[0] = self.body_height_weight * self._body_position()
        self.rewards[1] = self.body_orientation_weight * self._body_orientation_penalty()
        self.rewards[2] = self.po_violation_weight * self._check_rgc_violation()
        self.rewards[3] = self.stagnation_penalty_weight * self._stagnation_penalty()
        self.rewards[4] = self.stand_reward_weight * self._stand_stability_reward()
        self.rewards[5] = self.crouch_weight * self._crouch_reward()

        total_reward = self.rewards.sum()  #+ self._curriculum_learning_check()

        return total_reward

    def _body_position(self) -> float:
        """Penalizes if the body is too close to the ground."""
        return -1.0 if self.robot_states['b_pos'][1, 0] < 0.4 else 0.0

    def _body_orientation_penalty(self) -> float:
        """Penalizes the robot for excessive body pitch."""
        th = abs(self.robot_states['th'][0, 0])
        if th > self.th_threshold:
            overflow = th - self.th_threshold
            normalized_excess = min(overflow / (self.max_th - self.th_threshold), 1.0)
            return -normalized_excess
        return 0.0

    def _check_rgc_violation(self) -> float:
        """Penalizes if the RGC controller fails to solve the optimization problem."""
        # 'rcg_status' should be a key in your state dictionary
        rcg_status = self.robot_states.get('rcg_status')
        if rcg_status == 0:
            return -1.0
        elif rcg_status == -1:
            return -1.5
        return 0.0

    def _stagnation_penalty(self) -> float:
        """Penalizes the system for getting stuck in the same state."""
        if self.stagnation_metric == 0:
            return 0.0
        elif self.stagnation_metric < 0.1:
            return -1 * self.stagnation_metric
        else:
            return -1 / (1 + np.exp(-20 * (self.stagnation_metric - 0.2)))

    def _stand_stability_reward(self) -> float:
        """Curriculum Phase 1: Reward for standing upright and stable."""
        if self.curriculum_phase != 1:
            return 0.0

        r_z = self.robot_states['r_pos'][1, 0]
        th = abs(self.robot_states['th'][0, 0])
        current_mode = self.robot_states['cont_mode'][0, 0]
        valid_po = current_mode not in self.prohibited_actions

        com_ok = r_z > 0.75
        upright = th < 0.15

        if com_ok and upright and self.stagnation_steps > 0:
            self.phase1_success_steps += 1
            return 1.0
        elif com_ok and upright and valid_po:
            return 0.5
        elif valid_po:
            return 0.2
        return 0.0

    def _crouch_reward(self) -> float:
        """Curriculum Phase 2: Reward for crouching down."""
        if self.curriculum_phase != 2:
            return 0.0

        current_mode = self.robot_states['cont_mode'][0, 0]
        valid_po = current_mode not in self.prohibited_actions
        th = abs(self.robot_states['th'][0, 0])
        upright = th < 0.15

        if upright and valid_po:
            com_z = self.robot_states['r_pos'][1, 0]
            crouch_depth = self.standing_com_height - com_z
            reward = min(crouch_depth / self.max_com_drop, 1.0)
            return reward if reward > 0 else 0.0
        return 0.0

    def _curriculum_learning_check(self) -> float:
        """Checks for phase transitions in curriculum learning."""
        if (self.curriculum_phase == 1 and self.phase1_success_steps > self.phase1_success_steps_threshold):
            ic(f"End of phase 1 at step: {self.n_int}")
            self.curriculum_phase = 2
            self.standing_com_height = self.robot_states['r_pos'][1, 0]
            return 100.0  # Large bonus for completing the phase

        elif (self.curriculum_phase == 2 and self.stagnation_steps > 0 and self.robot_states['r_pos'][1, 0] < 0.65):
            ic(f"End of phase 2 at step: {self.n_int}")
            self.curriculum_phase = 3
            return 100.0  # Large bonus for completing the phase

        return 0.0

    def _update_stagnation_metric(self):
        """Update the stagnation metric based on state changes."""
        # Define the subset of states to monitor for stagnation
        current_state = np.hstack((
            self.robot_states['b_pos'].flatten(),
            self.robot_states['b_vel'].flatten(),
            self.robot_states['th'].flatten(),
            self.robot_states['dth'].flatten(),
        ))
        current_mode = self.robot_states['cont_mode']

        if self.prev_states is None:
            self.prev_states = current_state
            self.prev_mode = current_mode
            self.stagnation_metric = 0.0
            return

        delta = np.abs(current_state - self.prev_states)
        max_delta = np.max(delta)
        mode_unchanged = current_mode == self.prev_mode

        self.delta_history.append(max_delta)
        mu, std = np.mean(self.delta_history), np.std(self.delta_history)
        dyn_thresh = max(self.base_threshold, mu - self.multiplier * std)

        if mode_unchanged and max_delta < dyn_thresh:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0

        self.prev_states = current_state
        self.prev_mode = current_mode
        self.stagnation_metric = min(self.stagnation_steps / self.stagnation_threshold, 1.0)
