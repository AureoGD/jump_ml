from env_ga.multi_env import MultiGAEnv
from env_ga.switch_rule import SwitchRule


def main():
    n_robots = 5
    render = True

    # ✅ Best gene from GA
    best_threshold = 252  # Replace with your real value
    best_mode = 1  # Replace with your real value

    # Initialize environment
    env = MultiGAEnv(n_individuals=n_robots, render=render)

    # ✅ Use the same best rule for all robots
    rule_list = [SwitchRule([best_threshold], best_mode) for _ in range(n_robots)]

    env.reset_generation(rule_list)
    total_rewards = env.run_generation(max_steps=2000)

    print("\n=== Test Completed ===")
    for idx, reward in enumerate(total_rewards):
        print(f"Robot {idx}: Total Reward = {reward:.2f}")


if __name__ == "__main__":
    main()
