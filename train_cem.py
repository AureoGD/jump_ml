import os
import sys
import yaml
import numpy as np
import torch
from typing import Dict, Any, List

# --- Adjust these import paths based on your project structure ---
# Environment components
from env_ga.multi_env import MultiGAEnv
from env_ga.jump_states import _robot_states
from step_evaluator.step_evaluator import StepEvaluator

# Switch Rule (Policy)
from switch_rules.mlp_switch_rule import SimpleMLP

# ES Algorithm
from es_alg.cem_optimizer import CEMOptimizer, flatten_nn_parameters, unflatten_parameters_to_state_dict

# Utilities
from utils.training_logger import TrainingLogger
from utils.state_normalizer import StateNormalizer


def load_config_from_yaml(config_path="configs/env_config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # This assumes train_cem.py is in the project root where 'config' is a subdirectory
    actual_config_path = os.path.join(script_dir, config_path)

    if not os.path.exists(actual_config_path):
        print(f"ERROR: Configuration file not found at {actual_config_path}")
        raise FileNotFoundError(f"Config file not found: {actual_config_path}")

    with open(actual_config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {actual_config_path}")
    return config


if __name__ == "__main__":
    # --- 1. Load Configuration ---
    config = load_config_from_yaml()

    # --- 2. Extract Config Sections ---
    env_settings_cfg = config.get('environment', {})
    obs_setup_cfg = config.get('observation_setup', {})
    switch_rule_base_cfg = config.get('switch_rule_config', {})
    cem_hyperparams_cfg = config.get('training_config', {}).get('cem', {})
    logger_cfg = config.get('logger_config', {})
    normalization_config_from_yaml = config.get('normalization_config', {})

    # --- 3. Prepare Environment and NN Parameters ---
    observation_keys = obs_setup_cfg.get('keys', ['r_pos', 'r_vel', 'th', 'dth', 'q', 'dq'])
    temp_states_for_dim_calc = _robot_states()
    try:
        calculated_observation_dim = sum(temp_states_for_dim_calc[key].size for key in observation_keys)
    except KeyError as e:
        print(f"ERROR: Configured observation_key '{e.args[0]}' not found in jump_states.py. Exiting.")
        sys.exit(1)

    num_modes = switch_rule_base_cfg.get('num_modes', 3)

    # --- 4. State Normalizer Setup (Simplified) ---
    # The StateNormalizer is initialized directly with the config.
    # For 'online' types, it will use its default initial mean=0, var=1.
    # For 'static' types, it will use the min/max from the normalization_config_from_yaml.
    state_normalizer = StateNormalizer(
        observation_keys=observation_keys,
        config=normalization_config_from_yaml  # This config defines min/max for static, and type for online
    )
    print("StateNormalizer initialized using config (no pre-fitting in this script version).")

    # Kwargs for instantiating SwitchRule and StepEvaluator
    switch_rule_constructor_kwargs = {
        'observation_dim': calculated_observation_dim,
        'num_modes': num_modes,
        'state_normalizer': state_normalizer  # Pass the normalizer instance to the SwitchRule
    }
    # Add any other SwitchRule-specific args from switch_rule_base_cfg if they exist
    # e.g., switch_rule_constructor_kwargs.update(switch_rule_base_cfg.get('nn_architecture', {}))

    prepared_step_eval_kwargs = {
        'observation_keys': observation_keys
        # Add other StepEvaluator-specific args from step_eval_base_cfg if needed
    }

    # --- 5. Instantiate Main Training Environment ---
    population_size_cem = cem_hyperparams_cfg.get('population_size', 10)
    env = MultiGAEnv(
        n_individuals=population_size_cem,
        switch_rule_class=SimpleMLP,  # Your actual NN SwitchRule class
        step_evaluator_class=StepEvaluator,
        env_config=prepared_step_eval_kwargs,
        render=env_settings_cfg.get('render_simulation', False))

    # --- 6. Instantiate a Reference NN Model (for CEM structure and saving) ---
    # This model uses the same kwargs, including the normalizer instance
    reference_nn_model = SimpleMLP(**switch_rule_constructor_kwargs)
    nn_param_dim = flatten_nn_parameters(reference_nn_model).size
    print(f"SwitchRule NN parameter dimension: {nn_param_dim}")

    # --- 7. Instantiate CEM Optimizer ---
    cem_optimizer = CEMOptimizer(param_dim=nn_param_dim,
                                 population_size=population_size_cem,
                                 elite_fraction=cem_hyperparams_cfg.get('elite_fraction', 0.2),
                                 initial_std_dev=cem_hyperparams_cfg.get('initial_std_dev', 0.1),
                                 extra_noise_scale=cem_hyperparams_cfg.get('extra_noise_scale', 0.05),
                                 noise_decay_factor=cem_hyperparams_cfg.get('noise_decay_factor', 0.995),
                                 min_std_dev=cem_hyperparams_cfg.get('min_std_dev', 0.001))
    # Initialize CEM mean with the reference model's initial random weights
    cem_optimizer.set_initial_mean_params(reference_nn_model)

    # --- 8. Instantiate Training Logger ---
    logger = TrainingLogger(log_dir=logger_cfg.get('log_dir', "logs/cem_training"),
                            experiment_name=logger_cfg.get('experiment_name', "jumping_robot_cem"),
                            log_to_csv=logger_cfg.get('log_to_csv', True),
                            log_to_tensorboard=logger_cfg.get('log_to_tensorboard', True),
                            save_best_model=logger_cfg.get('save_best_model', True))
    logger.log_message(f"Starting CEM training for {logger_cfg.get('experiment_name', 'default_exp')}")
    logger.log_message(f"Full Config: {config}")

    # --- 9. Main CEM Training Loop ---
    num_generations = cem_hyperparams_cfg.get('num_generations', 100)
    max_steps_per_episode = env_settings_cfg.get('max_steps_per_episode', 1500)

    try:
        for gen in range(1, num_generations + 1):
            logger.log_message(f"--- Generation {gen}/{num_generations} ---")

            flat_weight_vectors_population = cem_optimizer.sample_population()

            rule_params_state_dicts = []
            for flat_weights in flat_weight_vectors_population:
                try:
                    state_dict = unflatten_parameters_to_state_dict(flat_weights, reference_nn_model)
                    rule_params_state_dicts.append(state_dict)
                except ValueError as e:
                    logger.log_message(f"Error unflattening weights for an individual: {e}")
                    continue

            if len(rule_params_state_dicts) != env.n_individuals:
                logger.log_message(
                    f"Warning: Population size for env.reset_generation ({len(rule_params_state_dicts)}) "
                    f"does not match expected ({env.n_individuals}) due to unflattening errors. "
                    "Adjusting or skipping generation if necessary.")
                # This situation should be handled carefully. If lists don't match, errors will occur.
                # For simplicity, if there's a mismatch, we might skip, or only use the valid ones
                # which would require adjusting how fitness_scores are matched.
                # A robust solution is to ensure unflattening always works or provides placeholder valid params.
                if not rule_params_state_dicts:  # If all failed
                    logger.log_message("All individuals failed unflattening. Skipping generation.")
                    continue
                # If some failed, we need to align flat_weight_vectors_population for cem_optimizer.update
                # For this simpler version, we assume if any fail, the whole generation might be problematic
                # or that `MultiGAEnv` should handle a potentially smaller list in reset_generation.
                # For now, this script proceeds, but this is a point of potential fragility.

            _ = env.reset_generation(rule_params_list=rule_params_state_dicts)
            fitness_scores = env.run_generation(max_steps=max_steps_per_episode)

            # Ensure fitness_scores correctly maps to the *original* flat_weight_vectors_population
            # This is tricky if some individuals were skipped during state_dict creation.
            # For now, assuming all `flat_weight_vectors` resulted in a fitness score.
            # A more robust way: `env.run_generation` could return fitness paired with original index or ID.
            if len(fitness_scores) != len(flat_weight_vectors_population):
                logger.log_message(
                    f"Warning: Mismatch between sampled population ({len(flat_weight_vectors_population)}) "
                    f"and obtained fitness scores ({len(fitness_scores)}). Check for errors.")
                # Attempt to use the shorter list for CEM update, which might not be ideal
                # This needs careful handling of which weights correspond to which scores.
                # For simplicity, let's assume they align for now if no errors in unflattening

            evaluated_population_for_cem = []
            # Only include individuals for whom we successfully got fitness, matching by index
            # This assumes that `fitness_scores` corresponds index-wise to `flat_weight_vectors_population`
            # and that `rule_params_state_dicts` was successfully used to set weights for each.
            num_successfully_evaluated = min(len(flat_weight_vectors_population), len(fitness_scores))
            for i in range(num_successfully_evaluated):
                evaluated_population_for_cem.append((flat_weight_vectors_population[i], fitness_scores[i]))

            if not evaluated_population_for_cem:
                logger.log_message("No individuals evaluated successfully for CEM update. Skipping CEM update.")
                continue

            cem_optimizer.update_distribution(evaluated_population_for_cem)

            logger.log_generation(generation=gen,
                                  evaluated_population=evaluated_population_for_cem,
                                  cem_optimizer=cem_optimizer,
                                  reference_model_for_saving=reference_nn_model)

    except KeyboardInterrupt:
        logger.log_message("Training interrupted by user.")
    except Exception as e:
        logger.log_message(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.log_message("Training loop finished or interrupted.")

        if cem_optimizer and reference_nn_model:  # Ensure they exist
            final_best_weights = cem_optimizer.get_best_params()
            if final_best_weights is not None:
                final_model_state_dict = unflatten_parameters_to_state_dict(final_best_weights, reference_nn_model)
                final_model_path = os.path.join(logger.models_save_dir, "cem_model_final_mean_params.pth")
                torch.save(final_model_state_dict, final_model_path)
                logger.log_message(f"Final best CEM mean weights saved to {final_model_path}")

        if 'logger' in locals() and logger is not None:  # Ensure logger was initialized
            logger.close()

    print("CEM training process complete.")
