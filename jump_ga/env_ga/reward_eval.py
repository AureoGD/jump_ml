# jump_ga/env_ga/reward_eval.py
from jump_ga.env_ga.reward_wrapper import RewardFcns


class RewardEvaluator:
    def __init__(self, robot_states):
        self.states = robot_states
        self.total_reward = 0.0
        self.terminated = False
        self.reward_evaluator = RewardFcns(self.states)

    def evaluate(self):
        """
        Evaluate reward and check if the episode should terminate.
        Example reward:
            - reward = CoM height
            - terminate if CoM height < 0.5 (fallen)
        """

        # Check if robot has fallen
        reward = self._rewards()
        self.total_reward += reward
        self.terminated = self._done()

        return reward, self.terminated

    def _rewards(self):
        return self.reward_evaluator.reward()

    def _done(self):
        if self.total_reward < -250:
            return True
        if self.reward_evaluator.n_int > 1000:
            return True
        if self.reward_evaluator.curriculum_phase > 2:
            True
        return False

    def reset(self):
        self.reward_evaluator.reset_variables()
        self.total_reward = 0.0
        self.terminated = False
