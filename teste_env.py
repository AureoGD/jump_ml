# train.py
import os
import sys
import yaml
import numpy as np
import torch
import multiprocessing
from typing import Dict, Any

# Adjust these import paths to match your project structure
from es_alg.cem_optimizer import CEMOptimizer, flatten_nn_parameters, unflatten_parameters_to_state_dict
from es_alg.worker import worker_init, run_worker_task # Uses your worker file
from step_evaluator.step_evaluator import StepEvaluator
from switch_rules.mlp_switch_rule import SimpleMLP
from utils.training_logger import TrainingLogger
from env_ga.jump_states import _robot_states as create_initial_robot_states
from utils.state_normalizer import StateNormalizer

def load_config(config_path="configs/env_config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at '{config_path}'")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return config

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Note: Multiprocessing start method could not be set.")

    # --- 1. Load Configuration ---
    config = load_config()
    cem_cfg = config.get('training_config', {}).get('cem', {})
    obs_cfg = config.get('observation_setup', {})
    switch_cfg = config.get('switch_rule_config', {})
    logger_cfg = config.get('logger_config', {})
    evaluator_cfg = config.get('evaluator_config', {})
    norm_cfg = config.get('normalization_config', {})

    # --- 2. Setup Logger and Prepare Configurations ---
    logger = TrainingLogger(**logger_cfg)
    
    observation_keys = obs_cfg.get('keys', [])
    temp_states = create_initial_robot_states()
    obs_dim = sum(temp_states[key].size for key in observation_keys)
    
    # Setup StateNormalizer (it will be passed to workers)
    state_normalizer = StateNormalizer(observation_keys=observation_keys, config=norm_cfg)
    # Note: We are still using the simpler approach of not pre-fitting the normalizer.
    # It will use static values from config and defaults for online types.
    
    switch_rule_kwargs = {
        'observation_dim': obs_dim,
        'num_modes': switch_cfg.get('num_modes'),
        'state_normalizer': state_normalizer
    }
    
    step_eval_kwargs = evaluator_cfg.copy()
    step_eval_kwargs['observation_keys'] = observation_keys
    config['step_eval_kwargs'] = step_eval_kwargs

    # --- 3. Instantiate Reference Model and CEM Optimizer ---
    reference_model = SimpleMLP(**switch_rule_kwargs)
    param_dim = flatten_nn_parameters(reference_model).size
    
    valid_cem_keys = ['population_size', 'elite_fraction', 'initial_std_dev', 'update_rule_type', 'elite_weighting_type', 'noise_decay_factor', 'min_std_dev', 'extra_noise_scale']
    cem_optimizer_kwargs = {key: cem_cfg[key] for key in valid_cem_keys if key in cem_cfg}
    cem_optimizer = CEMOptimizer(param_dim=param_dim, **cem_optimizer_kwargs)
    cem_optimizer.set_initial_mean_params(reference_model)

    # --- 4. Main Training Loop with a SINGLE, PERSISTENT Pool ---
    num_generations = cem_cfg.get('num_generations', 100)
    population_size = cem_optimizer.population_size
    num_workers = min(os.cpu_count() or 1, population_size)
    
    logger.log_message(f"Starting CEM training with {num_workers} persistent parallel workers.")
    
    # Arguments to initialize each worker process ONCE at the beginning.
    worker_init_args = (config, switch_rule_kwargs)

    # Create the pool of workers OUTSIDE the loop
    pool = multiprocessing.Pool(processes=num_workers, 
                                initializer=worker_init, 
                                initargs=worker_init_args)

    try:
        for gen in range(1, num_generations + 1):
            # a. Sample a population of NN weight vectors from the CEM
            population_params = cem_optimizer.sample_population()
            
            # b. Prepare tasks. Each task is just the unique data for that run.
            tasks = [(i, params) for i, params in enumerate(population_params)]

            # c. Map the tasks to the persistent worker pool.
            # The pool reuses the existing processes. No new workers are created here.
            results = pool.map(run_worker_task, tasks)

            # d. Process results
            results.sort(key=lambda x: x[0])
            fitness_scores = [score for _, score in results]
            
            # e. Update CEM and log
            evaluated_population = list(zip(population_params, fitness_scores))
            cem_optimizer.update_distribution(evaluated_population)
            395,
            logger.log_generation(
                generation=gen,
                evaluated_population=evaluated_population,
                cem_optimizer=cem_optimizer,
                reference_model_for_saving=reference_model
            )

    except KeyboardInterrupt:
        logger.log_message("Training interrupted by user.")
    finally:
        logger.log_message("Closing worker pool and saving final model...")
        
        # --- 5. Clean up the worker pool ---
        pool.close()
        pool.join()
        
        # ... (save final model and close logger) ...
        final_best_weights = cem_optimizer.get_best_params()
        final_model_state_dict = unflatten_parameters_to_state_dict(final_best_weights, reference_model)
        final_model_path = os.path.join(logger.models_save_dir, "cem_model_final_mean.pth")
        torch.save(final_model_state_dict, final_model_path)
        logger.log_message(f"Final CEM mean weights saved to {final_model_path}")
        logger.close()