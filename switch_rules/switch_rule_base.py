# jump_ga/switch_rule/switch_rule_base.py
import abc


class SwitchRuleBase(abc.ABC):
    """
    Abstract base class for all switching rules.
    """

    def __init__(self, observation_space=None, action_space=None):
        """
        Args:
            observation_space: (Optional) gym.spaces.Space object describing the input state.
            action_space: (Optional) gym.spaces.Space object describing the output action (mode).
        """
        self.observation_space = observation_space
        self.action_space = action_space  # Expected to be gym.spaces.Discrete for mode selection
        super().__init__()

    @abc.abstractmethod
    def get_mode(self, state) -> int:
        """
        Determines the MPC mode based on the current robot state.

        Args:
            state: The current state of the robot. The structure of this state
                   should be consistent with self.observation_space if provided.

        Returns:
            int: The selected MPC mode (as an integer).
        """
        pass

    def reset(self):
        """
        Resets any internal state of the rule (e.g., for recurrent NNs).
        Optional to implement if not needed.
        """
        pass
