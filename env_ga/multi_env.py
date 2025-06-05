# In file: env_ga/multi_env.py

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Any, Type, Tuple

# Assuming these modules are in the same directory or your Python path is set up correctly.
from .physics_world import PhysicsWorld
from .robot_model_ga import RobotModelGA
from .jump_states import _robot_states
# The actual classes for the rule and evaluator will be passed in,
# but importing them might be useful for type hinting.
# from ..switch_rules.switch_rule_base import SwitchRuleBase
# from .step_evaluator import StepEvaluator


class MultiGAEnv:
    """
    An environment to simulate a population of robot instances in parallel,
    designed for evolutionary algorithms and hybrid approaches.
    """

    def __init__(
            self,
            n_individuals: int,
            switch_rule_class: Type,
            step_evaluator_class: Type,
            env_config: Dict[str, Any] = None,  # <-- The single config dictionary
            render: bool = False):
        """
        Initializes the multi-robot environment from a single configuration dictionary.

        Args:
            n_individuals (int): The number of individuals in the population.
            switch_rule_class (Type): The class for the decision-making policy (e.g., an NN).
            step_evaluator_class (Type): The class that evaluates the outcome of each step.
            env_config (Dict): A configuration dictionary. Expected to contain 'observation_keys'.
                               Can optionally contain 'num_modes'.
            render (bool): If True, the simulation will be rendered.
        """
        if env_config is None:
            env_config = {}

        # --- Parse the single config and set defaults ---
        # 1. Get observation_keys, providing a default if missing.
        observation_keys = env_config.get('observation_keys', ['r_pos', 'r_vel', 'th', 'dth', 'q', 'dq'])
        if 'observation_keys' not in env_config:
            print(f"Warning: 'observation_keys' not in env_config. Using default: {observation_keys}")

        # 2. Dynamically calculate observation_dim based on the keys.
        # This enforces consistency and is a key benefit of your suggested change.
        temp_states = _robot_states()
        observation_dim = sum(temp_states[key].size for key in observation_keys)

        # 3. Get num_modes, providing a default if missing.
        num_modes = env_config.get('num_modes', 3)
        if 'num_modes' not in env_config:
            print(f"Warning: 'num_modes' not in env_config. Using default: {num_modes}")
        # --- End parsing and defaulting ---

        # The rest of the __init__ remains largely the same, but uses the derived parameters.
        self.n_individuals = n_individuals
        self.physics = PhysicsWorld(self.n_individuals, render=render)
        self.robots: List[Dict[str, Any]] = []

        # Prepare the specific kwargs dictionaries for the constructors
        switch_rule_kwargs = {'observation_dim': observation_dim, 'num_modes': num_modes}
        step_eval_kwargs = {'observation_keys': observation_keys}

        for i in range(self.n_individuals):
            robot_physical_states = _robot_states()
            model = RobotModelGA(robot_physical_states)
            rule = switch_rule_class(**switch_rule_kwargs)
            evaluator = step_evaluator_class(robot_physical_states, robot_model_ref=model, **step_eval_kwargs)

            self.robots.append({
                "id": i,
                "model": model,
                "rule": rule,
                "eval": evaluator,
                "states": robot_physical_states,
                "last_observation": None,
            })

        if self.n_individuals > 0:
            self.sim_per_rgc_step = int(self.robots[0]["model"].rgc_dt / self.robots[0]["model"].sim_dt)
        else:
            self.sim_per_rgc_step = 1

    def reset_generation(self, rule_params_list: List[Any]) -> List[np.ndarray]:
        """
        Resets all robot instances for a new generation, assigns new rule parameters,
        and returns the initial observations for the entire population.
        """
        if len(rule_params_list) != self.n_individuals:
            raise ValueError("Length of rule_params_list must match n_individuals.")

        initial_observations = []
        for i, params in enumerate(rule_params_list):
            robot = self.robots[i]

            if hasattr(robot["rule"], 'load_state_dict') and isinstance(params, dict):
                robot["rule"].load_state_dict(params)
            elif hasattr(robot["rule"], 'set_weights_from_vector') and isinstance(params, np.ndarray):
                robot["rule"].set_weights_from_vector(params)

            robot["model"].reset_variables()
            robot["eval"].reset()
            self.physics.reset_robot(i, robot["model"])

            initial_obs = robot["eval"].get_current_observation()
            robot["last_observation"] = initial_obs
            initial_observations.append(initial_obs)

        return initial_observations

    def _get_action_threaded(self, robot_dict: Dict[str, Any]) -> Tuple[int, int]:
        """Gets an action for a single robot based on its last known observation."""
        current_observation = robot_dict["last_observation"]
        mode = robot_dict["rule"].get_mode(current_observation, deterministic=True)
        robot_dict["model"].new_action(mode)
        return robot_dict["id"], mode

    def _evaluate_step_threaded(self, robot_dict: Dict[str, Any]) -> Tuple[int, float, bool, np.ndarray, Dict]:
        """Evaluates the outcome of the step for a single robot."""
        reward, done, info = robot_dict["eval"].evaluate()
        next_observation = robot_dict["eval"].get_current_observation()
        robot_dict["last_observation"] = next_observation
        return robot_dict["id"], reward, done, next_observation, info

    def run_generation(self, max_steps: int = 1500) -> List[float]:
        """
        Runs a full generation, evaluating all individuals in the population.
        """
        done_flags = [False] * self.n_individuals
        total_rewards = [0.0] * self.n_individuals

        active_robots = self.robots[:]

        for step_num in range(max_steps):
            if not active_robots:
                break

            with ThreadPoolExecutor() as executor:
                actions_results = list(executor.map(self._get_action_threaded, active_robots))
            actions_dict = dict(actions_results)

            for _ in range(self.sim_per_rgc_step):
                self.physics.step_all(self.robots)

            with ThreadPoolExecutor() as executor:
                step_results = list(executor.map(self._evaluate_step_threaded, active_robots))
            results_dict = {res[0]: res[1:] for res in step_results}

            new_active_robots_list = []
            for robot_dict in active_robots:
                robot_id = robot_dict["id"]
                reward, done, next_obs, info = results_dict[robot_id]

                total_rewards[robot_id] += reward
                done_flags[robot_id] = done

                # --- Log transition tuple for Replay Buffer (conceptual) ---
                # s_t = robot_dict["last_observation"]
                # a_t = actions_dict[robot_id]
                # r_t = reward
                # s_prime_t = next_obs
                # d_t = done
                # self.global_replay_buffer.add(s_t, a_t, r_t, s_prime_t, d_t)
                # -----------------------------------------------------------

                if not done:
                    new_active_robots_list.append(robot_dict)

            active_robots = new_active_robots_list

        return total_rewards
