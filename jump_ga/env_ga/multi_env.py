from concurrent.futures import ThreadPoolExecutor
from jump_ga.env_ga.physics_world import PhysicsWorld
from jump_ga.env_ga.robot_model_ga import RobotModelGA
from jump_ga.env_ga.reward_eval import RewardEvaluator
from jump_ga.env_ga.switch_rule import SwitchRule
from jump_modular_env.jump_model import RobotStates


class MultiGAEnv:
    def __init__(self, n_individuals, render=False):
        self.physics = PhysicsWorld(n_individuals, render=render)
        self.robots = []

        for i in range(n_individuals):
            states = RobotStates()
            model = RobotModelGA(states)
            rule = SwitchRule([100, 250])  # dummy thresholds
            eval_ = RewardEvaluator(states)

            model.set_switch_rule(rule)
            self.robots.append(
                {
                    "model": model,
                    "rule": rule,
                    "eval": eval_,
                    "states": states,
                }
            )

        # Determine control ratio from model
        model = self.robots[0]["model"]
        self.sim_per_rgc_step = int(model.rgc_dt / model.sim_dt)

    def reset_generation(self, rule_list):
        for i, rule in enumerate(rule_list):
            robot = self.robots[i]
            robot["rule"] = rule
            robot["model"].set_switch_rule(rule)
            robot["model"].reset_variables()
            robot["eval"].reset()
            self.physics.reset_robot(i, robot["model"])

    def run_generation(self, max_steps=1500):
        done_flags = [False] * len(self.robots)
        total_rewards = [0.0] * len(self.robots)

        for step in range(max_steps):
            # 1. Parallel: compute next mode + update qr (RGC)
            with ThreadPoolExecutor() as executor:
                executor.map(
                    self._update_action_threaded,
                    [(i, step) for i in range(len(self.robots)) if not done_flags[i]],
                )

            # 2. Step physics and apply torque at sim rate
            for _ in range(self.sim_per_rgc_step):
                self.physics.step_all(self.robots)

            # 3. Parallel: evaluate reward + check done
            with ThreadPoolExecutor() as executor:
                results = list(
                    executor.map(
                        self._evaluate_reward_threaded,
                        [i for i in range(len(self.robots)) if not done_flags[i]],
                    )
                )

            for idx, reward, done in results:
                total_rewards[idx] += reward
                done_flags[idx] = done_flags[idx] or done

            if all(done_flags):
                break

        return total_rewards

    def _update_action_threaded(self, args):
        i, step = args
        robot = self.robots[i]
        mode = robot["model"].evaluate_switch_rule(step)
        robot["model"].new_action(mode)

    def _evaluate_reward_threaded(self, i):
        robot = self.robots[i]
        reward, done = robot["eval"].evaluate()
        return i, reward, done
