# In: env_ga/step_evaluator.py
import numpy as np
from typing import List, Dict, Any, Tuple

# Assuming RewardFcns is defined in reward_wrapper
from .reward_wrapper import RewardFcns


class StepEvaluator:
    """
    Evaluates the outcome of a simulation step, including reward, termination, 
    and constructing the observation vector for the policy.
    """

    def __init__(self, robot_states: Dict[str, np.ndarray], observation_keys: List[str], robot_model_ref=None):
        """
        Args:
            robot_states (Dict[str, np.ndarray]): Reference to the robot's state dictionary.
            observation_keys (List[str]): A list of keys from the robot_states dict
                                         that define the desired observation vector.
            robot_model_ref: Optional reference to RobotModelGA, potentially needed for
                             complex reward calculations.
        """
        self.states = robot_states
        self.robot_model_ref = robot_model_ref

        # This is the crucial change: the evaluator now knows which states
        # constitute an "observation" for the policy.
        self.observation_keys = observation_keys

        self.total_reward = 0.0
        self.terminated = False
        self.reward_calculator = RewardFcns(self.states, robot_model_ref=self.robot_model_ref)

    def get_current_observation(self) -> np.ndarray:
        """
        Constructs the observation vector by concatenating the state values
        specified by self.observation_keys. This is the definitive source
        for what the policy/agent sees.
        """
        obs_parts = []
        for key in self.observation_keys:
            if key in self.states:
                obs_parts.append(self.states[key].flatten())
            else:
                # Provide a warning if a key for the observation is not found in the state dictionary
                print(f"Warning: Observation key '{key}' not found in robot states dict. This may cause an error.")

        if not obs_parts:
            raise ValueError("Observation could not be constructed. No valid keys were provided or found.")

        observation = np.concatenate(obs_parts).astype(np.float32)

        # --- NORMALIZATION ---
        # A StateNormalizer instance would be used here if provided.
        # e.g., if self.normalizer: return self.normalizer.transform(observation)

        return observation

    def evaluate(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates reward and checks for termination based on the current state.
        
        Returns:
            A tuple containing (reward, done, info).
        """
        reward = self._rewards()
        self.total_reward += reward
        self.terminated = self._done()

        # Info dictionary for auxiliary data and diagnostics
        info = {
            'total_reward': self.total_reward,
            'episode_step': self.reward_calculator.n_int,
            'curriculum_phase': self.reward_calculator.curriculum_phase
        }

        return reward, self.terminated, info

    def _rewards(self) -> float:
        """Calculates the reward for the current step."""
        return self.reward_calculator.reward()

    def _done(self) -> bool:
        """Checks for episode termination conditions."""
        if self.total_reward < -250:
            return True
        if self.reward_calculator.n_int > 1000:
            return True
        if self.reward_calculator.curriculum_phase > 2:
            return True
        return False

    def reset(self):
        """Resets the evaluator's internal state for a new episode."""
        self.reward_calculator.reset_variables()
        self.total_reward = 0.0
        self.terminated = False

    def calculate_fitness(self, states):
        return 0
