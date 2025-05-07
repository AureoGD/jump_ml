import numpy as np
from icecream import ic
from collections import deque


class RewardFcns:
    def __init__(self, model_states, debug=False):
        self.robot_states = model_states

        # self.max_transitions = hor_lenght

        self.rewards = np.zeros((20, 1), dtype=np.double)

        # states variables
        self.transition_history = 0
        self.foot_contact_state = 0
        self.stagnation_metric = 0
        self.total_modes_changes = 0
        self.n_int = 0

        """ reward weights  """
        self.body_height_weight = 2
        self.body_orietation_weight = 3.5
        self.po_violation_weight = 1.5
        self.short_term_mode_transition_weight = 1
        self.long_term_mode_transition_weight = 1
        self.invalid_po_weight = 1.5
        self.min_mode_duration_weight = 1.2
        self.stagnation_penalty_weight = 10.0
        self.stand_reward_weight = 2
        self.crouch_weight = 1.0

        """ rewards auxiliary constants"""
        # Body orientaton penality
        self.th_threshold = 0.65  # Angle (in rad) where penalty begins
        self.max_th = 1.2  # Maximum angle for full penalty (overflow capped)

        # Invalid PO penalty
        self.invalid_po_grace_period = 15

        # minimal mode duration
        self.mode_min_duration = {
            0: 16,
            1: 25,
            2: 20,
            3: 10,
            4: 12,
            5: 12,
            6: 16,
            7: 18,
        }

        self.prohibited_actions = [6, 7]

        self.phase1_success_steps_threshold = 50

        self.max_com_drop = 0.2

        self.base_threshold = 0.002
        self.multiplier = 1.5
        self.stagnation_threshold = 150

        """ rewards auxiliary variables """
        self.reset_variables()

    def reset_variables(self):
        self.n_int = 0
        self.invalid_po_steps = 0
        self.mode_duration = 0
        self.last_action = -1
        self.total_mode_changes = 0
        self.standing_com_height = 0
        self.curriculum_phase = 1
        self.stagnation_steps = 0
        self.phase1_success_steps = 0
        self.prev_states = None
        self.delta_history = deque(maxlen=30)

    def reward(self):
        """Evaluate all rewards terms and sum the values

        Returns:
            float: the sum of the rewards
        """
        self.n_int += 1
        self._update_stagnation_metric()

        self.rewards[:] = 0

        # Penalize if the body is near of the ground
        self.rewards[0] = self.body_height_weight * self._body_position()

        # Penalize the body orietation
        self.rewards[1] = self.body_orietation_weight * self._body_orientation_penalty()

        # Penalizes if the the PO used is no capable to be solved
        self.rewards[2] = self.po_violation_weight * self._check_rgc_violation()

        # Penalize stagnation
        self.rewards[3] = self.stagnation_penalty_weight * self._stagnation_penalty()

        # Reward for the system stability in early stages
        self.rewards[4] = self.stand_reward_weight * self._stand_stability_reward()

        # Reward for crouch in the second phase
        self.rewards[5] = self.crouch_weight * self._crouch_reward()

        reward = self.rewards.sum() + self._curriculum_learning_check()

        return reward

    def _body_position(self):
        """
        Penalie if the body is almost in the ground
        """
        if self.robot_states.b_pos[1, 0] < 0.4:
            return -1
        else:
            return 0

    def _body_orientation_penalty(self):
        """
        Penalize the robot for exceeding the allowable body pitch orientation (in radians).

        The penalty starts when the absolute value of the body orientation angle (th)
        exceeds a defined threshold (0.8 rad). The penalty scales linearly between 0
        and -1 as the orientation approaches a maximum tolerated value (e.g., 1.5 rad).

        Returns:
            float: A negative reward (penalty) in the range [-1, 0], or 0 if within threshold.
        """
        th = abs(self.robot_states.th[0, 0])

        if th > self.th_threshold:
            overflow = th - self.th_threshold
            normalized_excess = min(overflow / (self.max_th - self.th_threshold), 1.0)
            return -normalized_excess
        else:
            return 0

    def _check_rgc_violation(self):
        if self.robot_states.rcg_status == 0:
            return -1
        elif self.robot_states.rcg_status == -1:
            return -1.5
        else:
            return 0

    def _stagnation_penalty(self):
        """
        Pezalizes the stagnation of the system. If the PO not change and the states are almost the same, penalizes using a picewise function
        """
        if self.stagnation_metric == 0:
            return 0.0
        elif self.stagnation_metric < 0.1:
            return -1 * self.stagnation_metric
        else:
            return -1 / (1 + np.exp(-20 * (self.stagnation_metric - 0.2)))

    def _stand_stability_reward(self):
        """
        Curriculum Phase 1: Reward the agent for standing upright, with a valid PO,
        low linear and angular velocity, and a minimum CoM height.

        Reward scales from 0 to 1.
        """
        if self.curriculum_phase != 1:
            return 0
        r_z = self.robot_states.r_pos[1, 0]
        th = abs(self.robot_states.th[0, 0])
        valid_po = self.robot_states.mode not in self.prohibited_actions

        com_ok = r_z > 0.75
        upright = th < 0.15

        # if com_ok and upright and valid_po and low_motion:
        if com_ok and upright and self.stagnation_steps > 0:
            self.phase1_success_steps += 1
            return 1.0
        elif com_ok and upright and valid_po:
            return 0.5
        elif valid_po:
            return 0.2
        return 0

    def _crouch_reward(self):
        """
        Reward the robot for crouching (i.e., lowering CoM from the standing position),
        simulating the potential energy storage phase of a spring.

        Returns:
            float: Reward in range [0, 1], where 1 is max crouch.
        """
        # Make sure phase 1 was completed and reference height is stored
        if self.curriculum_phase != 2:
            return 0

        valid_po = self.robot_states.mode not in self.prohibited_actions

        th = abs(self.robot_states.th[0, 0])
        upright = th < 0.15

        if upright and valid_po:
            com_z = self.robot_states.r_pos[1, 0]
            crouch_depth = self.standing_com_height - com_z
            reward = min(crouch_depth / self.max_com_drop, 1.0)
            return reward
        else:
            return 0

    def _curriculum_learning_check(self):
        if (
            self.curriculum_phase == 1
            and self.phase1_success_steps > self.phase1_success_steps_threshold
        ):
            ic(f"End of phase 1: {self.n_int}")
            self.curriculum_phase = 2
            self.standing_com_height = self.robot_states.r_pos[1, 0]
            return 100
        elif (
            self.curriculum_phase == 2
            and self.stagnation_steps > 0
            and self.robot_states.r_pos[1, 0] < 0.65
        ):
            ic(f"End of phase 2: {self.n_int}")
            self.curriculum_phase = 3
            return 100

        return 0

    def _update_stagnation_metric(self):
        """
        Update the stagnation metric based on max delta of current state.
        Returns a float in [0, 1], where 1 = completely stagnant.
        """

        current_state = np.hstack(
            (
                self.robot_states.b_pos.flatten(),
                self.robot_states.b_vel.flatten(),
                self.robot_states.th.flatten(),
                self.robot_states.dth.flatten(),
            )
        )

        current_mode = self.robot_states.mode

        if self.prev_states is None:
            self.prev_states = current_state
            self.prev_mode = current_mode
            self.stagnation_metric = 0.0
            return self.stagnation_metric

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

        return min(self.stagnation_steps / self.stagnation_threshold, 1.0)
