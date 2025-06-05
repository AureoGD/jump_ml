# test_multi_ga_env.py
import numpy as np
import torch  # If your SwitchRule is a PyTorch NN
import os
import yaml  # For loading config from YAML file

# Adjust these import paths based on your actual directory structure
from env_ga.multi_env import MultiGAEnv
from env_ga.jump_states import _robot_states
from step_evaluator.step_evaluator import StepEvaluator
from switch_rules.switch_rule_base import SwitchRuleBase
from utils.state_normalizer import StateNormalizer


class DummySwitchRule(SwitchRuleBase, torch.nn.Module):

    def __init__(self, observation_dim: int, num_modes: int, **kwargs):  # Added **kwargs for flexibility
        SwitchRuleBase.__init__(self)
        torch.nn.Module.__init__(self)
        self.observation_dim = observation_dim
        self.num_modes = num_modes
        # Example layer: a real NN would have more complex architecture
        self.dummy_linear = torch.nn.Linear(observation_dim, num_modes)
        print(f"DummySwitchRule initialized with obs_dim={observation_dim}, num_modes={num_modes}")

    def get_mode(self, state: np.ndarray, deterministic: bool = False) -> int:
        # For this dummy rule, we don't use the state to make a decision
        return np.random.randint(0, self.num_modes)

    def load_state_dict(self, state_dict, strict=True):
        # This method would be called by your ES algorithm to set the NN weights
        print(f"DummySwitchRule: load_state_dict called (params: {list(state_dict.keys()) if state_dict else 'None'})")
        # For a real NN: super().load_state_dict(state_dict, strict=strict)
        # For this dummy, we just acknowledge the call.
        # If state_dict is not empty, a real NN might try to load it.
        # If it's empty, a real NN would use its random initialization.
        pass

    def set_weights_from_vector(self, weights_vector: np.ndarray):
        # Alternative way to set weights if your ES provides a flat vector
        print(f"DummySwitchRule: set_weights_from_vector called with vector of shape {weights_vector.shape}")
        # Logic to unflatten and set weights for self.dummy_linear would go here
        pass


def load_config_from_yaml(config_path="configs/env_config.yaml"):
    """Loads configuration from a YAML file."""
    # Construct the absolute path to the config file relative to this script's location
    # This assumes teste.py is in the root of 'jump_ml-es_strategies'
    # and 'config' is a subdirectory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If teste.py is not at project root, you might need to adjust path, e.g.:
    # project_root = os.path.abspath(os.path.join(script_dir, "..")) # If teste.py is one level down
    # actual_config_path = os.path.join(project_root, config_path)
    actual_config_path = os.path.join(script_dir, config_path)

    if not os.path.exists(actual_config_path):
        print(f"ERROR: Configuration file not found at {actual_config_path}")
        print("Please ensure 'config/env_config.yaml' exists relative to the project root or where this script is run.")
        return None

    with open(actual_config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {actual_config_path}")
    return config


if __name__ == "__main__":
    print("Starting MultiGAEnv test with unified config from YAML...")

    # --- 1. Load the configuration from YAML ---
    config = load_config_from_yaml()  # Uses the default path "config/env_config.yaml"

    if config is None:
        exit()  # Exit if config couldn't be loaded

    # --- 2. Extract and prepare arguments for MultiGAEnv's constructor ---
    env_settings_cfg = config.get('environment', {})
    obs_setup_cfg = config.get('observation_setup', {})
    switch_rule_base_cfg = config.get('switch_rule_config', {})
    # step_eval_base_cfg = config.get('step_evaluator_config', {}) # If you add specific keys here

    n_individuals = env_settings_cfg.get('n_individuals', 1)
    render_simulation = env_settings_cfg.get('render_simulation', False)
    max_episode_steps = env_settings_cfg.get('max_steps_per_episode', 1500)

    observation_keys = obs_setup_cfg.get('keys', ['r_pos', 'r_vel'])
    # shape_depth = obs_setup_cfg.get('shape_depth', 0) # For future use with state history

    temp_states_for_dim_calc = _robot_states()
    try:
        calculated_observation_dim = sum(temp_states_for_dim_calc[key].size for key in observation_keys)
    except KeyError as e:
        print(f"ERROR: A key in observation_keys ('{e.args[0]}') from your YAML config "
              "was not found in initial robot states.")
        print("Available state keys from create_initial_robot_states():", list(temp_states_for_dim_calc.keys()))
        exit()

    prepared_switch_rule_kwargs = {
        'observation_dim': calculated_observation_dim,
        'num_modes': switch_rule_base_cfg.get('num_modes', 3)
        # Add other SwitchRule-specific args from switch_rule_base_cfg if needed
    }

    prepared_step_eval_kwargs = {
        'observation_keys': observation_keys
        # Add other StepEvaluator-specific args from step_eval_base_cfg if needed
    }

    state_normalizer = StateNormalizer(observation_keys=observation_keys, config=config.get('normalization_config'))

    # --- 3. Instantiate MultiGAEnv ---
    print(f"Instantiating MultiGAEnv with n_individuals={n_individuals}...")
    print(f"  SwitchRule kwargs: {prepared_switch_rule_kwargs}")
    print(f"  StepEvaluator kwargs: {prepared_step_eval_kwargs}")
    try:
        env = MultiGAEnv(n_individuals=n_individuals,
                         switch_rule_class=DummySwitchRule,
                         step_evaluator_class=StepEvaluator,
                         env_config=prepared_step_eval_kwargs,
                         render=render_simulation)
        print("MultiGAEnv instantiated successfully.")
    except Exception as e:
        print(f"Error during MultiGAEnv instantiation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 4. Prepare dummy rule parameters for reset_generation ---
    dummy_rule_params_list = [{} for _ in range(n_individuals)]

    print("Calling reset_generation...")
    try:
        initial_observations = env.reset_generation(rule_params_list=dummy_rule_params_list)
        print(f"reset_generation called successfully. Shape of first initial obs: {initial_observations[0].shape}")
        if initial_observations[0].shape[0] != calculated_observation_dim:
            print(f"ERROR: Observation dimension mismatch! Expected {calculated_observation_dim}, "
                  f"Got {initial_observations[0].shape[0]}")
    except Exception as e:
        print(f"Error during reset_generation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 5. Run for a few steps ---
    print(f"Calling run_generation for {max_episode_steps} steps...")
    try:
        total_rewards = env.run_generation(max_steps=max_episode_steps)
        print(f"run_generation completed. Total rewards: {total_rewards}")
    except Exception as e:
        print(f"Error during run_generation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    print("Test script finished. Check for rendering window and any errors.")

    if render_simulation:
        print("Render window should be open. Close it manually or press Ctrl+C in the terminal to exit script.")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Exiting test...")
        finally:
            if hasattr(env, 'close'):
                env.close()
            elif hasattr(env.physics, 'disconnect'):
                env.physics.disconnect()
            # It's good practice to ensure PyBullet disconnects if not handled by env.close()
            # This might require direct access to the pybullet instance if physics world doesn't expose disconnect
            import pybullet as p
            if p.isConnected():
                p.disconnect()
