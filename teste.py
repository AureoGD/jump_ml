import numpy as np
import torch  # If your SwitchRule is a PyTorch NN
import os
import yaml  # For loading config from YAML file
import cProfile  # For profiling
import pstats  # For printing profiling stats
import io  # For capturing pstats output as string

# Adjust these import paths based on your actual directory structure
from env_ga.multi_env import MultiGAEnv
from env_ga.jump_states import _robot_states  # Corrected import
from step_evaluator.step_evaluator import StepEvaluator
from switch_rules.switch_rule_base import SwitchRuleBase
from utils.state_normalizer import StateNormalizer  # Assuming this is your normalizer class


class DummySwitchRule(SwitchRuleBase, torch.nn.Module):

    def __init__(self,
                 observation_dim: int,
                 num_modes: int,
                 state_normalizer=None,
                 **kwargs):  # Added state_normalizer and **kwargs
        SwitchRuleBase.__init__(self)
        torch.nn.Module.__init__(self)
        self.observation_dim = observation_dim
        self.num_modes = num_modes
        self.state_normalizer = state_normalizer  # Store the normalizer
        self.dummy_linear = torch.nn.Linear(observation_dim, num_modes)
        print(
            f"DummySwitchRule initialized with obs_dim={observation_dim}, num_modes={num_modes}, normalizer={'Yes' if state_normalizer else 'No'}"
        )

    def get_mode(self, state: np.ndarray, deterministic: bool = False) -> int:
        # In a real NN, you'd normalize then pass to forward
        # normalized_state = state
        # if self.state_normalizer:
        #     normalized_state = self.state_normalizer.normalize(state)
        # state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0)
        # # logits = self.dummy_linear(state_tensor)
        # return np.random.randint(0, self.num_modes)
        return 0

    def load_state_dict(self, state_dict, strict=True):
        print(f"DummySwitchRule: load_state_dict called (params: {list(state_dict.keys()) if state_dict else 'None'})")
        # For a real NN, you'd call: super().load_state_dict(state_dict, strict=strict)
        pass

    def set_weights_from_vector(self, weights_vector: np.ndarray):
        print(f"DummySwitchRule: set_weights_from_vector called with vector of shape {weights_vector.shape}")
        pass


def load_config_from_yaml(config_path="configs/env_config.yaml"):
    """Loads configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    actual_config_path = os.path.join(script_dir, config_path)

    if not os.path.exists(actual_config_path):
        print(f"ERROR: Configuration file not found at {actual_config_path}")
        print("Please ensure 'config/env_config.yaml' exists relative to this script's location.")
        return None

    with open(actual_config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {actual_config_path}")
    return config


def main_test_logic():
    """Main logic of the test script, to be profiled."""
    print("Starting MultiGAEnv test with unified config from YAML...")

    config = load_config_from_yaml()
    if config is None:
        exit()

    env_settings_cfg = config.get('environment', {})
    obs_setup_cfg = config.get('observation_setup', {})
    switch_rule_base_cfg = config.get('switch_rule_config', {})
    normalization_cfg = config.get('normalization_config', {})

    n_individuals = env_settings_cfg.get('n_individuals', 1)
    render_simulation = env_settings_cfg.get('render_simulation', False)
    max_episode_steps = env_settings_cfg.get('max_steps_per_episode', 50)  # Reduced for profiling

    observation_keys = obs_setup_cfg.get('keys', ['r_pos', 'r_vel'])

    temp_states_for_dim_calc = _robot_states()
    try:
        calculated_observation_dim = sum(temp_states_for_dim_calc[key].size for key in observation_keys)
    except KeyError as e:
        print(f"ERROR: A key in observation_keys ('{e.args[0]}') from your YAML config "
              "was not found in initial robot states.")
        print("Available state keys:", list(temp_states_for_dim_calc.keys()))
        exit()

    # Initialize StateNormalizer
    state_normalizer_instance = StateNormalizer(observation_keys=observation_keys, config=normalization_cfg)
    # In a real scenario, you might fit it here if not using purely static/default online:
    # sample_data = ... collect some initial data ...
    # state_normalizer_instance.fit(sample_data)
    print("StateNormalizer initialized (using YAML config for static, defaults for online if not fitted).")

    prepared_switch_rule_kwargs = {
        'observation_dim': calculated_observation_dim,
        'num_modes': switch_rule_base_cfg.get('num_modes', 3),
        'state_normalizer': state_normalizer_instance  # Pass the normalizer
    }

    prepared_step_eval_kwargs = {
        'observation_keys': observation_keys
        # Pass normalizer to StepEvaluator if it also needs to normalize for its own get_observation
        # 'state_normalizer': state_normalizer_instance
    }

    print(f"Instantiating MultiGAEnv with n_individuals={n_individuals}...")
    print(f"  SwitchRule kwargs: {{'observation_dim': {prepared_switch_rule_kwargs['observation_dim']}, "
          f"'num_modes': {prepared_switch_rule_kwargs['num_modes']}, "
          f"'state_normalizer': {'Yes' if prepared_switch_rule_kwargs['state_normalizer'] else 'No'} }}")
    print(f"  StepEvaluator kwargs: {prepared_step_eval_kwargs}")
    try:
        env = MultiGAEnv(n_individuals=1,
                         switch_rule_class=DummySwitchRule,
                         step_evaluator_class=StepEvaluator,
                         env_config=prepared_step_eval_kwargs,
                         render=True)
        print("MultiGAEnv instantiated successfully.")
    except Exception as e:
        print(f"Error during MultiGAEnv instantiation: {e}")
        import traceback
        traceback.print_exc()
        exit()

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

    print(f"Calling run_generation for {max_episode_steps} steps...")
    try:
        total_rewards = env.run_generation(max_steps=max_episode_steps)
        print(f"run_generation completed. Total rewards: {total_rewards}")
    except Exception as e:
        print(f"Error during run_generation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    print("Test script main logic finished.")
    return env  # Return env for cleanup if needed


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    env_instance = main_test_logic()  # Run the main part of the script

    profiler.disable()

    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE  # Can also be 'TIME', 'CALLS'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(30)  # Print top 30 functions
    print("\n--- cProfile Output ---")
    print(s.getvalue())
    print("--- End cProfile Output ---\n")

    print("Test script finished. Check for rendering window and any errors.")

    if env_instance and env_instance.physics.render:
        print("Render window should be open. Close it manually or press Ctrl+C in the terminal to exit script.")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Exiting test...")
        finally:
            if hasattr(env_instance, 'close'):
                env_instance.close()
            elif hasattr(env_instance.physics, 'disconnect'):
                env_instance.physics.disconnect()
            else:  # Fallback to direct pybullet disconnect if needed
                try:
                    import pybullet as p
                    if p.isConnected():
                        p.disconnect()
                except ImportError:
                    pass  # Pybullet not available, or already disconnected
