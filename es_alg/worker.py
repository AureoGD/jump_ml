# es_algs/worker.py
import os
import numpy as np
import torch
from typing import Dict, Any, Tuple

# --- This section assumes your project files are accessible in the child process ---
from env_ga.individual import Individual
from switch_rules.mlp_switch_rule import SimpleMLP
from step_evaluator.step_evaluator import StepEvaluator
from es_alg.cem_optimizer import unflatten_parameters_to_state_dict

# --- Global variables for the persistent worker objects ---
sim_instance: Individual = None
evaluator: StepEvaluator = None
switch_rule: SimpleMLP = None


def worker_init(config: Dict[str, Any], switch_rule_kwargs: Dict[str, Any]):
    """
    Initializer function for each worker process. Runs ONCE per worker.
    Creates persistent instances of the simulation and the evaluator.
    """
    global sim_instance, evaluator, switch_rule
    
    process_id = os.getpid()
    print(f"[Proc {process_id}] Initializing worker...")

    # 1. Create the SwitchRule (policy) for this worker
    switch_rule = SimpleMLP(**switch_rule_kwargs)

    # 2. Create the persistent Individual simulation environment
    sim_instance = Individual(
        individual_id=process_id,
        config=config
    )
    
    # 3. Create the persistent StepEvaluator
    # The evaluator now uses the persistent sim_instance's state dictionary
    evaluator_cfg = config.get('evaluator_config', {})
    evaluator_kwargs = config.get('step_eval_kwargs', {}) # Get kwargs for StepEvaluator
    evaluator = StepEvaluator(
        robot_states=sim_instance.robot_states, # Pass the sim's state dict
        **evaluator_kwargs
    )
    
    print(f"[Proc {process_id}] Worker initialized successfully.")


def run_worker_task(task_args: Tuple[int, np.ndarray]) -> Tuple[int, float]:
    """
    This is the main task function. It REUSES the persistent Individual and Evaluator objects.
    """
    global sim_instance, evaluator
    if sim_instance is None or evaluator is None:
        raise RuntimeError("Worker not initialized correctly.")

    task_id, nn_params_flat = task_args
    
    try:
        # 1. Load the new weights into the existing SwitchRule NN
        state_dict = unflatten_parameters_to_state_dict(nn_params_flat, switch_rule)
        switch_rule.load_state_dict(state_dict)

        # 2. Run the episode loop
        full_state_history = []
        
        # --- CORRECTED DATA FLOW ---
        # a. Reset the simulation and get the initial observation vector
        sim_instance.reset()
        current_observation = evaluator.get_current_observation()
        max_steps = sim_instance.config.get('environment', {}).get("max_steps_per_episode", 1000)
        step = 0
        for step in range(max_steps):
            # b. The policy uses the np.ndarray observation to get an action
            action = switch_rule.get_mode(current_observation)
            
            # c. The simulation executes one step
            step_history = sim_instance.one_step(action)
            full_state_history.extend(step_history)
            
            # d. Check for early termination
            if evaluator._done():
                break
            
            # e. Get the NEXT observation vector for the next loop iteration
            current_observation = evaluator.get_current_observation()
        # --- END OF CORRECTED DATA FLOW ---

        # 3. Calculate fitness from the full trajectory
        fitness = evaluator.calculate_fitness(full_state_history)
        
        return task_id, fitness

    except Exception as e:
        print(f"[Proc {os.getpid()}, Task {task_id}] ERROR in worker task: {e}")
        import traceback
        traceback.print_exc()
        return task_id, -float('inf')