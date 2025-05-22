import numpy as np
from collections import deque
from icecream import ic
from .state_normalizer_v2 import StateNormalizer
from .reward_wrapper import RewardFcns
from stagnation_classifier.cnn_stagnation_wrapper import StagnationClassifier


class MLWrapper:

    def __init__(self, _robot_states, debug=True, policy_type="cnn"):
        if debug:
            ic.enable()
        else:
            ic.disable()

        self.robot_states = _robot_states
        self.policy_type = policy_type

        # Gym variables
        self.OBS_LOW_VALUE = -1
        self.OBS_HIGH_VALUE = 1
        self.NUM_OBS_STATES_P = 15  # r, dr, th, dth, q, dq, qr
        self.NUM_OBS_STATES_NP = 11  # tau, modeion, trans_history, contact_, stagnation_metric, b, db

        self.NP = 30  # Size of the past steates buffer
        self.NF = 10  # Size of the future states buffer
        self.NUM_MODES = 3

        self.reward_fcns = RewardFcns(debug=True, model_states=_robot_states, hor_lenght=self.NP)

        self.state_normalizer = StateNormalizer(
            22,
            self.NUM_MODES,
            self.NP,
            3,
        )

        # Buffer of the last N actions
        self.actions = deque([-1] * self.NP, maxlen=self.NP)

        self.pred_st = np.zeros((self.NUM_OBS_STATES_P, self.NF))

        # deque of the states
        zero_pred = [np.zeros(self.NUM_OBS_STATES_P) for _ in range(self.NP)]
        zero_nonpred = [np.zeros(self.NUM_OBS_STATES_NP) for _ in range(self.NP)]

        self.predictable_history = deque(zero_pred, maxlen=self.NP)
        self.nonpredictable_history = deque(zero_nonpred, maxlen=self.NP)

        self.base_past = deque([np.zeros(10) for _ in range(self.NP)], maxlen=self.NP)  # [r, dr, th, dth, b, db,]
        self.joint_past = deque([np.zeros(12) for _ in range(self.NP)], maxlen=self.NP)  # [q, dq, qr, tau]
        self.comp_states = deque([np.zeros(5) for _ in range(self.NP)],
                                 maxlen=self.NP)  # [stag, phase, trans_h, foot_st, sucess]

        self.dt = 0.01

        # curent number of simulation interation/step
        self.inter = 0

        # How many mode changes
        self.transition_history = 0
        self.total_modes_changes = 0  # episode
        self.foot_contact_state = 0

        # # stagnation metric
        # CLASSIFIER_MODEL_PATH = "./stagnation_classifier/models/cnn_classifier_IV.pth"
        CLASSIFIER_MODEL_PATH = "./stagnation_classifier/models/cnn_regressor.pth"
        self.classifier = StagnationClassifier(CLASSIFIER_MODEL_PATH)
        # self.classifier = StagnationClassifier(CLASSIFIER_MODEL_PATH)

        self.stagnation_metric = 0.0
        self.stagnation_steps = 0
        self.stagnation_threshold = 25
        self.base_threshold = 0.002
        self.multiplier = 1.5
        self.delta_history = deque(maxlen=self.NP)
        self.prev_states = None
        self.prev_mode = None

        self.n_jumps = 0
        self.total_mode_changes = 0

        self.episode_reward = 0
        self.min_reward = -250
        self.success_rate = 0
        self.stag_value = 0

    def end_of_step(self):
        """
        After each step, get observation, reward and done. Also increment the numer of interation
        """
        self.inter += 1
        self.transition_history = self._check_transition()
        self.foot_contact_state = self._check_contact_mode()
        self.stagnation_metric = self._update_stagnation_metric()
        self.success_rate = self._compute_success_from_cost()

        reward = self._reward()
        obs = self._observation()

        terminated = self._done()

        return obs, reward, terminated

    def observation(self):
        self.transition_history = self._check_transition()
        self.foot_contact_state = self._check_contact_mode()
        self.stagnation_metric = self._update_stagnation_metric()
        return self._observation()

    def _observation(self):
        # Raw system state stacking
        base_states = np.vstack((
            self.robot_states.r_pos,
            self.robot_states.r_vel,
            self.robot_states.th,
            self.robot_states.dth,
            self.robot_states.b_vel,
            self.robot_states.b_pos,
        ))  # (10, 1)

        joint_states = np.vstack((
            self.robot_states.q,
            self.robot_states.dq,
            self.robot_states.qr,
            self.robot_states.tau,
        ))  # (12, 1)

        comp_state = np.vstack((
            np.array([[self.stagnation_metric]]),
            np.array([[self.reward_fcns.curriculum_phase]]),
            self.robot_states.mode,
            self.foot_contact_state,
            np.array([[self.success_rate]]),
        ))  # shape (5, 1)

        # Normalize
        states = np.vstack((base_states, joint_states))
        norm_states = self.state_normalizer.normalize(states)  # shape (22, 1)

        base_vec = norm_states[:10].flatten()  # (10,)
        joint_vec = norm_states[10:].flatten()  # (12,)

        # Append to history
        self.base_past.appendleft(base_vec)
        self.joint_past.appendleft(joint_vec)
        self.comp_states.appendleft(comp_state.flatten())  # (5,)

        # Normalize predicted states
        for t in range(self.NF):
            self.pred_st[:, t] = self.state_normalizer.normalize(self.pred_st[:, t])

        return {
            "base_past": np.stack(self.base_past, axis=1).astype(np.float32),  # (10, NP)
            "base_future": self.pred_st[:10, :].astype(np.float32),  # (10, NF)
            "joint_past": np.stack(self.joint_past, axis=1).astype(np.float32),  # (12, NP)
            "joint_future": self.pred_st[10:, :].astype(np.float32),  # (12, NF)
            "comp_past": np.stack(self.comp_states, axis=1).astype(np.float32),  # (5, NP)
        }

    def _reward(self):
        self.reward_fcns.update_variables(
            n_int=self.inter,
            foot_state=self.foot_contact_state,
            stag_metric=self.stagnation_metric,
            transtion_hist=self.transition_history,
            total_mod_chan=self.total_modes_changes,
            stag_steps=self.stagnation_steps,
        )
        reward = self.reward_fcns.reward()
        self.episode_reward += reward
        return reward

    def _done(self):
        """
        Return if the the episode is done.
        """
        if self.episode_reward <= self.min_reward:
            return True
        if self.reward_fcns.curriculum_phase * 400 <= self.inter:
            return True
        if self.robot_states.b_pos[1, 0] < 0.2:
            return True
        if self.reward_fcns.curriculum_phase > 2:
            return True

        # if self.inter >= 1000:
        #     return True

        # otherwise continue
        return False

    def _check_transition(self):
        """
        Evaluate the total valid transitions and how many transtions happen in the last NP interations
        """
        current_action = int(self.robot_states.mode)
        self.actions.appendleft(current_action)

        if (self.actions[0] != self.actions[1]) and (self.actions[1] != -1):
            self.total_modes_changes += 1

        return sum(1 for i in range(len(self.actions) - 1, 0, -1)
                   if self.actions[i] != self.actions[i - 1] and self.actions[i] != -1 and self.actions[i - 1] != -1)

    def _check_contact_mode(self):
        """
        Evaluate the contacte configuratio:

        heel (*2) | toe | value
          0       |  0  |   0   no contact
          0       |  1  |   1   only toe
          1       |  0  |   2   only heel
          1       |  1  |   3   heel and toe

        """
        return (self.robot_states.toe_cont + self.robot_states.heel_cont * 2) / 3

    def _update_stagnation_metric(self):
        """
        Update the stagnation metric based on max delta of current state.
        Returns a float in [0, 1], where 1 = completely stagnant.
        """

        self.stag_value = self.classifier.predict(self.base_past, self.joint_past, self.comp_states)

        if self.stag_value >= 0.5:
            self.stagnation_steps += 1
        elif self.stagnation_steps > 0 and self.stag_value < 0.5:
            self.stagnation_steps -= 1

        if self.stagnation_steps > self.stagnation_threshold:
            stag_over = self.stagnation_steps - self.base_threshold
            return min(stag_over / self.stagnation_threshold, 1.0)
        else:
            return 0

    def _compute_success_from_cost(self, alpha=0.1):
        """
        Compute success metric from QP cost.
        Args:
            cost (float): The QP solver optimal cost.
            alpha (float): Scaling factor (tune based on experiments).
        Returns:
            float: Success in [0, 1].
        """
        success = np.exp(-alpha * self.robot_states.j_val[0, 0])
        return success

    def update_pred_states(self, _pred_states):
        """
        Comy the predict states from MPC controller
        [r, dr, th, dth, q, dq, qr]
        """
        self.pred_st = _pred_states.copy(order="C")

    def reset_vars(self):
        self.episode_reward = 0
        self.inter = 0
        self.total_modes_changes = 0

        self.actions = deque([-1] * self.NP, maxlen=self.NP)

        self.stagnation_metric = 0.0
        self.stagnation_steps = 0
        self.prev_states = None
        self.prev_mode = None
        self.delta_history.clear()

        self.reward_fcns.reset_variables()
