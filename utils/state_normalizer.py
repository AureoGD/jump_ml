import numpy as np
from typing import List, Dict, Any


class StateNormalizer:
    """
    Handles normalization of a state vector defined by a list of keys.
    Supports hybrid normalization:
    - 'static': Min-max scaling to a specified range (default [-1, 1]) using fixed bounds.
    - 'online': Standardization using a running mean and variance.
    """

    def __init__(self, observation_keys: List[str], config: Dict[str, Any]):
        """
        Args:
            observation_keys (List[str]): The ordered list of keys that defines the observation vector.
            config (Dict): A configuration dictionary specifying the normalization type and
                           bounds for each key.
                           Example for a key 'feature_name':
                           'feature_name': {'type': 'static', 'min': 0.0, 'max': 1.0, 'target_min': -1, 'target_max': 1}
                           'feature_name': {'type': 'static', 'min': [0.0, -1.0], 'max': [1.0, 1.0]} # target defaults to [-1,1]
                           'feature_name': {'type': 'online'}
        """
        self.keys = observation_keys
        self.config = config
        self.n_features_total = 0

        self.mean: Dict[str, np.ndarray] = {}
        self.var: Dict[str, np.ndarray] = {}  # Stores variance

        self.alpha: Dict[str, np.ndarray] = {}  # For static scaling
        self.beta: Dict[str, np.ndarray] = {}  # For static scaling

        self._initialize_params_and_dims()

    def _get_feature_shape(self, key: str) -> tuple:
        """Helper to get the shape of a feature from its initial state definition."""
        # This relies on _robot_states being accessible and consistent
        from env_ga.jump_states import _robot_states  # Adjust import as necessary
        initial_states = _robot_states()
        if key not in initial_states:
            raise ValueError(f"Key '{key}' for normalization not found in initial robot states definition.")
        return initial_states[key].shape

    def _initialize_params_and_dims(self):
        """
        Initializes parameters based on config and calculates total feature dimension.
        Pre-calculates scaling factors for 'static' normalization.
        """
        current_total_dim = 0
        for key in self.keys:
            feature_shape = self._get_feature_shape(key)
            feature_size = np.prod(feature_shape)  # Total number of elements for this feature
            current_total_dim += feature_size

            key_config = self.config.get(key, {})
            norm_type = key_config.get('type')

            if norm_type == 'static':
                min_val_config = key_config.get('min', -1.0)
                max_val_config = key_config.get('max', 1.0)
                target_min = key_config.get('target_min', -1.0)
                target_max = key_config.get('target_max', 1.0)

                # --- Process min_val ---
                # Convert config value to a NumPy array first
                min_val_np = np.array(min_val_config, dtype=np.float32)

                # If config value is scalar (0-D array or single element)
                if min_val_np.size == 1:
                    # Broadcast this scalar value to match the feature_shape
                    min_val = np.full(feature_shape, min_val_np.item(), dtype=np.float32)
                # If config value is an array, its shape must match the feature_shape
                elif min_val_np.shape[0] == feature_shape[0]:
                    min_val = min_val_np
                else:
                    raise ValueError(
                        f"Shape mismatch for static 'min' of key '{key}'. "
                        f"Feature shape from states: {feature_shape}, min_val shape from config: {min_val_np.shape}. "
                        f"Configured min_val: {min_val_config}")

                # --- Process max_val (similarly) ---
                max_val_np = np.array(max_val_config, dtype=np.float32)

                if max_val_np.size == 1:
                    max_val = np.full(feature_shape, max_val_np.item(), dtype=np.float32)
                elif max_val_np.shape[0] == feature_shape[0]:
                    max_val = max_val_np
                else:
                    raise ValueError(
                        f"Shape mismatch for static 'max' of key '{key}'. "
                        f"Feature shape from states: {feature_shape}, max_val shape from config: {max_val_np.shape}. "
                        f"Configured max_val: {max_val_config}")

                # Now, min_val and max_val are guaranteed to be NumPy arrays with shape = feature_shape

                input_range = max_val - min_val
                target_range = float(target_max) - float(target_min)  # Ensure target_range is scalar for division
                # or make target_min/max arrays if needed

                is_zero_range = (np.abs(input_range) < 1e-8)

                # Alpha and Beta will also have the same shape as the feature
                self.alpha[key] = np.where(is_zero_range, np.zeros_like(input_range),
                                           target_range / (input_range + 1e-8))
                self.beta[key] = np.where(
                    is_zero_range,
                    np.full(feature_shape, float(target_min)),  # Ensure beta matches shape
                    float(target_min) - self.alpha[key] * min_val)

                self.mean[key] = np.zeros(feature_shape, dtype=np.float32)
                self.var[key] = np.ones(feature_shape, dtype=np.float32)

                # Calculate scaling parameters (alpha, beta) for:
                # normalized_value = alpha * original_value + beta
                # This maps [min_val, max_val] to [target_min, target_max]
                input_range = max_val - min_val
                target_range = target_max - target_min

                # Handle constant input features (input_range is zero)
                is_zero_range = (np.abs(input_range) < 1e-8)

                self.alpha[key] = np.where(
                    is_zero_range,
                    np.zeros_like(input_range),  # alpha = 0 if input is constant
                    target_range / (input_range + 1e-8))  # Add epsilon for safety
                self.beta[key] = np.where(
                    is_zero_range,
                    target_min,  # if input is constant, map it to target_min
                    target_min - self.alpha[key] * min_val)

                # Initialize mean/var even for static, might be useful for inspection
                self.mean[key] = np.zeros(feature_shape, dtype=np.float32)
                self.var[key] = np.ones(feature_shape, dtype=np.float32)

            elif norm_type == 'online':
                self.mean[key] = np.zeros(feature_shape, dtype=np.float32)
                self.var[key] = np.ones(feature_shape, dtype=np.float32)  # Initialize variance to 1
                # Alpha/beta are not strictly needed for online, but can be placeholders
                self.alpha[key] = np.ones(feature_shape, dtype=np.float32)
                self.beta[key] = np.zeros(feature_shape, dtype=np.float32)
            else:  # No normalization or unknown type for this key
                print(f"Warning: No or unknown normalization type for key '{key}'. It will not be normalized.")
                self.mean[key] = np.zeros(feature_shape, dtype=np.float32)
                self.var[key] = np.ones(feature_shape, dtype=np.float32)
                self.alpha[key] = np.ones(feature_shape, dtype=np.float32)  # Pass through (y = 1*x + 0)
                self.beta[key] = np.zeros(feature_shape, dtype=np.float32)

        self.n_features_total = current_total_dim

    def update(self, state_dict: Dict[str, np.ndarray], momentum=0.001):
        """
        Updates the running mean and variance for 'online' features.
        Args:
            state_dict (Dict): The full robot_states dictionary for the current step.
            momentum (float): The momentum for the exponential moving average.
        """
        for key in self.keys:
            if key in state_dict and self.config.get(key, {}).get('type') == 'online':
                feature_values = state_dict[key].astype(np.float32).reshape(self.mean[key].shape)

                delta = feature_values - self.mean[key]
                self.mean[key] += momentum * delta
                self.var[key] += momentum * (np.square(delta) - self.var[key])
                self.var[key] = np.maximum(self.var[key], 1e-8)  # Ensure variance is non-negative

    def normalize(self, observation_vector: np.ndarray) -> np.ndarray:
        """
        Normalizes a complete observation vector feature by feature according to its key.
        """
        if observation_vector.size != self.n_features_total:
            raise ValueError(f"Observation vector size {observation_vector.size} "
                             f"does not match expected total feature size {self.n_features_total}.")

        x_norm = np.zeros_like(observation_vector, dtype=np.float32)
        current_idx = 0

        for key in self.keys:
            feature_shape = self.mean[key].shape  # Use shape from initialized params
            feature_size = np.prod(feature_shape)

            feature_slice = observation_vector[current_idx:current_idx + feature_size]
            reshaped_feature_slice = feature_slice.reshape(feature_shape)  # Reshape to original feature dims

            norm_type = self.config.get(key, {}).get('type')

            if norm_type == 'online':
                normalized_slice = (reshaped_feature_slice - self.mean[key]) / np.sqrt(self.var[key] + 1e-8)
            elif norm_type == 'static':
                normalized_slice = self.alpha[key] @ reshaped_feature_slice + self.beta[key]
            else:
                normalized_slice = reshaped_feature_slice  # Pass through if no normalization

            x_norm[current_idx:current_idx + feature_size] = normalized_slice.flatten()
            current_idx += feature_size

        return x_norm

    def save_stats(self, filepath: str):
        np.savez(filepath,
                 keys=np.array(self.keys, dtype=object),
                 config=self.config,
                 mean=self.mean,
                 var=self.var,
                 alpha=self.alpha,
                 beta=self.beta,
                 n_features_total=self.n_features_total)
        print(f"Normalization stats saved to {filepath}")

    def load_stats(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.keys = list(data['keys'])
        self.config = data['config'].item()
        self.mean = data['mean'].item()
        self.var = data['var'].item()
        self.alpha = data['alpha'].item()
        self.beta = data['beta'].item()
        self.n_features_total = data['n_features_total'].item()
        # _initialize_params_and_dims might not be needed if all relevant state is loaded
        # However, for static params (alpha, beta) it's good to re-ensure they match the loaded config.
        # For simplicity, ensure loaded config is used if _initialize_params_and_dims is complex to re-run.
        # Or, ensure that save/load stores everything needed to bypass _initialize_params_and_dims fully.
        # For this version, we assume loading these dicts is sufficient.
        print(f"Normalization stats loaded from {filepath}")
