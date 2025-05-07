import numpy as np
from collections import deque
from icecream import ic
from .state_normalizer import StateNormalizer
from .reward_wrapper import RewardFcns


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
        self.NUM_OBS_STATES_NP = (
            11  # tau, modeion, trans_history, contact_, stagnation_metric, b, db
        )

        self.NP = 30  # Size of the past steates buffer
        self.NF = 10  # Size of the future states buffer
        self.NUM_MODES = 3

        self.reward_fcns = RewardFcns(
            debug=True, model_states=_robot_states, hor_lenght=self.NP
        )

        self.state_normalizer = StateNormalizer(
            self.NUM_OBS_STATES_NP + self.NUM_OBS_STATES_P,
            self.NUM_MODES,
            self.NP,
            3,
        )

        # Buffer of the last N actions
        self.actions = deque([-1] * self.NP, maxlen=self.NP)

        self.pred_st = np.zeros((self.NUM_OBS_STATES_P, self.NF))

        # deque of the states
        self.predictable_history = deque(maxlen=self.NP)
        self.nonpredictable_history = deque(maxlen=self.NP)

        self.dt = 0.01

        # curent number of simulation interation/step
        self.inter = 0

        # How many mode changes
        self.transition_history = 0
        self.total_modes_changes = 0  # episode
        self.foot_contact_state = 0

        # stagnation metric
        self.stagnation_metric = 0.0
        self.stagnation_steps = 0
        self.stagnation_threshold = 150
        self.base_threshold = 0.002
        self.multiplier = 1.5
        self.delta_history = deque(maxlen=self.NP)
        self.prev_states = None
        self.prev_mode = None

        self.n_jumps = 0
        self.total_mode_changes = 0

        self.episode_reward = 0
        self.min_reward = -250

    def end_of_step(self):
        """
        After each step, get observation, reward and done. Also increment the numer of interation
        """
        self.inter += 1
        obs = self._observation()
        reward = self._reward()
        terminated = self._done()

        return obs, reward, terminated

    def observation(self):
        return self._observation()

    def _observation(self):
        """
        This funtion is responsable to evalaute the observed states of the system.
        First some of the states are evaluated;
        Then the actual state vector is created, and normalized;
        The predict states from the RGC-MPC are normalized too;
        Then the observation dictionary is create as:
            [past_pred_st, actual_pred_sts states, pred_st]
            [past_non_pred_st, actual_non__pred_st]
        """
        # append the new action to the list of last N actions and evaluate the history of the transation
        self.transition_history = self._check_transition()
        self.foot_contact_state = self._check_contact_mode()
        self.stagnation_metric = self._update_stagnation_metric()

        states = np.vstack(
            (
                self.robot_states.r_vel,
                self.robot_states.r_pos,
                self.robot_states.th,
                self.robot_states.dth,
                self.robot_states.q,
                self.robot_states.dq,
                self.robot_states.qr,
                self.robot_states.tau,
                self.robot_states.mode,
                self.transition_history,
                self.foot_contact_state,
                np.array([[self.stagnation_metric]]),
                self.robot_states.b_vel,
                self.robot_states.b_pos,
            ),
        )

        norm_states = self.state_normalizer.normalize(states)

        for t in range(self.NF):
            self.pred_st[:, t] = self.state_normalizer.normalize(self.pred_st[:, t])

        # First 15: predictable features (r_vel, r_pos, th, dth, q, dq, qr)
        self.predictable_history.appendleft(norm_states[0:15])
        # Last 7: non-predictable features (tau, mode, transition, contact, stagnation)
        self.nonpredictable_history.appendleft(norm_states[15:])

        # shape: (obs_dim, N_past + N_future)
        predictable_obs = np.hstack(list(self.predictable_history) + [self.pred_st])

        # shape: (obs_dim, N_past)
        # predictable_obs = np.hstack(list(self.predictable_history))

        if self.policy_type == "dnn":
            predictable_obs = predictable_obs.flatten()
            nonpredictable_obs = np.hstack(list(self.nonpredictable_history)).flatten()
        else:
            predictable_obs = predictable_obs
            nonpredictable_obs = np.hstack(list(self.nonpredictable_history))

        return {
            "predictable": predictable_obs.astype(np.float32),
            "nonpredictable": nonpredictable_obs.astype(np.float32),
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
        # if self.episode_reward <= self.min_reward:
        #     return True
        # if self.reward_fcns.curriculum_phase * 750 <= self.inter:
        #     return True
        if self.robot_states.b_pos[1, 0] < 0.2:
            return True

        if self.inter >= 5000:
            return True

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

        return sum(
            1
            for i in range(len(self.actions) - 1, 0, -1)
            if self.actions[i] != self.actions[i - 1]
            and self.actions[i] != -1
            and self.actions[i - 1] != -1
        )

    def _check_contact_mode(self):
        """
        Evaluate the contacte configuratio:

        heel (*2) | toe | value
          0       |  0  |   0   no contact
          0       |  1  |   1   only toe
          1       |  0  |   2   only heel
          1       |  1  |   3   heel and toe

        """
        return self.robot_states.toe_cont + self.robot_states.heel_cont * 2

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
            self.delta_history.clear()
            self.stagnation_steps = 0

        self.prev_states = current_state
        self.prev_mode = current_mode

        if self.stagnation_steps > self.stagnation_threshold:
            stag_over = self.stagnation_steps - self.base_threshold
            return min(stag_over / self.stagnation_threshold, 1.0)
        else:
            return 0

    def update_pred_states(self, _pred_states):
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
