import numpy as np
from icecream import ic
import random


class RewardFcns:

    def __init__(self, model_states, hor_lenght, debug=True):
        self.robot_states = model_states

        self.max_transitions = hor_lenght

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
        self.short_term_mode_transition_weight = 5
        self.long_term_mode_transition_weight = 5
        self.invalid_po_weight = 1.5
        self.min_mode_duration_weight = 5
        self.stagnation_penalty_weight = 5.0
        self.stand_reward_weight = 2
        self.crouch_weight = 2.0
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

        self.phase1_success_steps_threshold = 2

        self.max_com_drop = 0.2
        """ rewards auxiliary variables """
        self.reset_variables()

    def reset_variables(self):
        self.invalid_po_steps = 0
        self.mode_duration = 0
        self.last_action = -1
        self.total_mode_changes = 0
        self.standing_com_height = 0
        self.stagnation_steps = 0
        self.phase1_success_steps = 0

        beta = random.random()

        if beta > 0.5:
            self.curriculum_phase = 2
            self.standing_com_height = 0.85
        else:
            self.curriculum_phase = 1
            self.standing_com_height = 0.85

        self.curriculum_phase = 1

        ic(f"Start phaase: {self.curriculum_phase}")
        # self.curriculum_phase = 1

    def update_variables(self, n_int, transtion_hist, foot_state, stag_metric, total_mod_chan, stag_steps):
        self.n_int = n_int
        self.transition_history = transtion_hist
        self.foot_contact_state = foot_state
        self.stagnation_metric = stag_metric
        self.total_mode_changes = total_mod_chan
        self.stagnation_steps = stag_steps

    def reward(self):
        """Evaluate all rewards terms and sum the values

        Returns:
            float: the sum of the rewards
        """
        self.rewards[:] = 0

        # Penalize if the body is near of the ground
        self.rewards[0] = self.body_height_weight * self._body_position()

        # Penalize the body orietation
        self.rewards[1] = self.body_orietation_weight * self._body_orientation_penalty()

        # Penalize hight PO changes in short term
        self.rewards[2] = (self.short_term_mode_transition_weight * self._short_term_mode_transition_penalty())

        # Penalize hight PO changes in long term
        self.rewards[3] = (self.long_term_mode_transition_weight * self._long_term_mode_transition_penalty())

        # Penalize if the PO not stay for a minimum period
        self.rewards[4] = (self.min_mode_duration_weight * self._minimum_mode_duration_penalty())

        # Penalizes invalid PO choices due to the robot configuration: contact/non-contact
        self.rewards[5] = self.invalid_po_weight * self._invalid_po_penalty()

        # Penalizes if the the PO used is no capable to be solved
        self.rewards[6] = self.po_violation_weight * self._check_rgc_violation()

        # Penalize stagnation
        self.rewards[7] = self.stagnation_penalty_weight * self._stagnation_penalty()

        # Reward for the system stability in early stages
        self.rewards[8] = self.stand_reward_weight * self._stand_stability_reward()

        # Reward for crouch in the second phase
        self.rewards[9] = self.crouch_weight * self._crouch_reward()

        # Reward for choose an aproprieate model in a specifi phase
        self.rewards[10] = self._preferred_mode_bonus()

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

    def _short_term_mode_transition_penalty(self):
        """
        Penalizes recent excessive changes in PO (short-term jitteriness).
        Uses transition history over last N steps. Keeps in [-1, 1].
        """
        if self.transition_history > 1:
            return -min(self.transition_history / self.max_transitions, 1.0)
        return 0.0

    def _long_term_mode_transition_penalty(self):
        """
        Penalizes high total number of action changes over time.
        Helps prevent instability over the episode. Returns [-1, 0].
        """

        change_ratio = self.total_mode_changes / self.n_int
        return -min(change_ratio, 1.0)

    def _invalid_po_penalty(self):
        """
        Penalize staying in a prohibited PO (mode) while foot is in contact.
        The penalty decreases gradually and is capped at -1. Keeps reward in [-1, 0].
        """
        # TODO: reward logic can be added to give positive values for use the right PO if in contact
        current_action = int(self.robot_states.mode)
        if self.foot_contact_state != 0 and current_action in [6, 7]:
            self.invalid_po_steps += 1
        elif self.foot_contact_state == 0 and current_action in [0, 1, 2, 3, 4, 5]:
            self.invalid_po_steps += 1
        else:
            self.invalid_po_steps = 0

        if self.invalid_po_steps > self.invalid_po_grace_period:
            max_steps = self.invalid_po_grace_period * 3
            over_steps = self.invalid_po_steps - self.invalid_po_grace_period
            normalized_penalty = -min(over_steps / (max_steps - self.invalid_po_grace_period), 1.0)
            return normalized_penalty
        return 0

    def _minimum_mode_duration_penalty(self):
        """
        Penalizes if the current mode has not been maintained for at least `min_steps` steps.
        Optionally scales the penalty in [-1, 0] based on how close the agent is to meeting the requirement.

        Args:
            min_steps (int): Minimum number of steps required in the same mode before avoiding penalty.
            scaled (bool): Whether to scale penalty smoothly from -1 to 0.

        Returns:
            float: A penalty in [-1.0, 0.0]
        """
        current_action = int(self.robot_states.mode)
        if current_action == -1 or self.last_action == -1:
            return 0.0  # Ignore uninitialized states

        if current_action == self.last_action:
            self.mode_duration += 1
        else:
            self.mode_duration = 0

        min_steps = self.mode_min_duration.get(current_action, 10)

        if self.mode_duration < min_steps:
            reward = -1.0 * (1.0 - self.mode_duration / min_steps)
        else:
            reward = 0

        self.last_action = current_action

        return reward

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
        else:
            return -1 * self.stagnation_metric

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
        if (self.curriculum_phase == 1 and self.phase1_success_steps > self.phase1_success_steps_threshold):
            ic(f"End of phase 1: {self.n_int}")
            self.curriculum_phase = 2
            self.standing_com_height = self.robot_states.r_pos[1, 0]
            return 100
        elif (self.curriculum_phase == 2 and self.stagnation_steps > 0 and self.robot_states.r_pos[1, 0] < 0.65):
            ic(f"End of phase 2: {self.n_int}")
            self.curriculum_phase = 3
            return 100

        return 0

    def _preferred_mode_bonus(self):
        """
        During curriculum phase 1, give a small bonus for using empirically preferred modes.
        Helps encourage stable behaviors early in training.

        Returns:
            float: A small positive reward (e.g., 0.5) or 0 otherwise.
        """
        if self.curriculum_phase == 1 and self.robot_states.mode in [0, 1]:
            return 1.0
        if self.curriculum_phase == 1 and self.robot_states.mode == 2:
            return -0.5
        if self.curriculum_phase == 2 and self.robot_states.mode == 2:
            return 1.0
        if self.curriculum_phase == 2 and self.robot_states.mode in [0, 1]:
            return -1.0
        return 0
