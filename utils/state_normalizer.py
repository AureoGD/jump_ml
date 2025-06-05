# In: preprocessing/state_normalizer.py
import numpy as np
from typing import List, Dict, Any


class StateNormalizer:
    """
    Handles normalization of a state vector defined by a list of keys.
    Supports hybrid normalization:
    - 'static': Min-max scaling to [-1, 1] using fixed bounds.
    - 'online': Standardization using a running mean and variance.
    """

    def __init__(self, observation_keys: List[str], config: Dict[str, Any]):
        """
        Args:
            observation_keys (List[str]): The ordered list of keys that defines the observation vector.
            config (Dict): A configuration dictionary specifying the normalization type and
                           bounds for each key.
        """
        self.keys = observation_keys
        self.config = config
        self.n_features = len(self.keys)

        # Dictionaries to store normalization parameters, keyed by the state name
        self.mean = {key: 0.0 for key in self.keys}
        self.var = {key: 1.0 for key in self.keys}  # Storing variance, not std dev

        self.alpha = {}  # For static scaling
        self.beta = {}  # For static scaling

        self._setup_static_params()

    def _setup_static_params(self):
        """Pre-calculates scaling factors for features with static normalization."""
        for key in self.keys:
            key_config = self.config.get(key, {})
            if key_config.get('type') == 'static':
                min_val = key_config.get('min', -1.0)
                max_val = key_config.get('max', 1.0)
                # Formula to scale to [-1, 1]
                self.alpha[key] = 2.0 / (max_val - min_val + 1e-8)
                self.beta[key] = -(max_val + min_val) / (max_val - min_val + 1e-8)

    def update(self, state_dict: Dict[str, np.ndarray], momentum=0.001):
        """
        Updates the running mean and variance for 'online' features.

        Args:
            state_dict (Dict): The full robot_states dictionary.
            momentum (float): The momentum for the exponential moving average.
        """
        for key in self.keys:
            if self.config.get(key, {}).get('type') == 'online':
                # This assumes the state value is a scalar or can be treated as one
                value = state_dict[key].item()  # .item() for single-element arrays
                delta = value - self.mean[key]
                self.mean[key] += momentum * delta
                # Update variance with the squared difference
                self.var[key] += momentum * (delta**2 - self.var[key])

    def normalize(self, observation_vector: np.ndarray) -> np.ndarray:
        """
        Normalizes a complete observation vector feature by feature.
        
        Args:
            observation_vector (np.ndarray): The vector constructed by StepEvaluator.
        
        Returns:
            np.ndarray: The normalized vector.
        """
        x_norm = np.zeros_like(observation_vector)
        current_idx = 0
        for key in self.keys:
            key_config = self.config.get(key, {})
            # Get the size of this feature (e.g., a vector like 'q' has size 3)
            feature_size = self.mean[key].size if isinstance(self.mean[key], np.ndarray) else 1

            # Extract the slice corresponding to this key from the input vector
            feature_slice = observation_vector[current_idx:current_idx + feature_size]

            if key_config.get('type') == 'online':
                # Standardization: (x - mean) / sqrt(variance)
                x_norm[current_idx : current_idx + feature_size] = \
                    (feature_slice - self.mean[key]) / np.sqrt(self.var[key] + 1e-8)

            elif key_config.get('type') == 'static':
                # Min-Max Scaling: alpha * x + beta
                x_norm[current_idx : current_idx + feature_size] = \
                    self.alpha[key] * feature_slice + self.beta[key]

            else:  # If type is not specified, do not normalize
                x_norm[current_idx:current_idx + feature_size] = feature_slice

            current_idx += feature_size

        return x_norm

    # save() and load() methods would now save/load the dictionaries:
    # self.mean, self.var, self.alpha, self.beta, self.keys, self.config
