# teste.py
import yaml
import time
import os
import numpy as np
from random import Random

# --- Adjust imports based on your final directory structure ---
from env_ga.individual import Individual  # Your new Individual class
from switch_rules.switch_rule_base import SwitchRuleBase  # The base class


# For this test, we create a simple dummy rule that cycles through actions.
# In the real training script, this would be your CEM-evolved NN.
class CycleSwitchRule(SwitchRuleBase):

    def __init__(self, num_modes, steps_per_mode=100):
        super().__init__()
        self.num_modes = num_modes
        self.steps_per_mode = steps_per_mode
        self.current_step = 0

    def get_mode(self, state: np.ndarray, deterministic: bool = False) -> int:
        # Simple logic: cycle through actions 0, 1, 2, ...
        mode = (self.current_step // self.steps_per_mode) % self.num_modes
        self.current_step += 1
        return mode

    def reset(self):
        self.current_step = 0


def load_config(config_path="configs/test_config.yaml"):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return config


if __name__ == "__main__":
    print("--- Starting Individual Simulation Test ---")

    # 1. Load Configuration
    config = load_config()

    # 2. Instantiate the Switch Rule and the Individual
    # In a real scenario, the SwitchRule would be your NN. Here, we use a simple dummy.
    switch_rule_cfg = config.get('switch_rule_config', {})
    dummy_rule = CycleSwitchRule(num_modes=switch_rule_cfg.get('num_modes', 3))

    # The Individual class manages its own internal components based on the config
    individual_sim = Individual(individual_id=0, config=config)

    try:
        # --- Run two full episodes to test reset functionality ---
        action = 0
        for episode in range(10):
            print(f"\n--- Starting Episode {episode + 1} ---")
            individual_sim.reset()

            print(action)

            # The evaluator will create the first observation during reset
            # current_obs = individual_sim.evaluator.get_current_observation()

            max_steps = config.get('environment', {}).get('max_steps_per_episode', 300)

            for step in range(max_steps):
                step_states_history = individual_sim.one_step(action)

            print(f"--- Episode {episode + 1} Finished ---")
            time.sleep(1)  # Pause for a moment before resetting
            # action += 1
            # if action == 3:
            #     action = 0

    except Exception as e:
        print(f"\nAn error occurred during the simulation test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure the simulation is cleanly disconnected
        print("\nClosing simulation.")
        individual_sim.close()
