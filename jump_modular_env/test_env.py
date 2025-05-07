import numpy as np
from jump_modular_env.jump_env import JumperEnv


def run_test():
    # Create the environment
    env = JumperEnv(render=True, policy_type="mlp", debug=True)

    # Reset environment
    obs, info = env.reset()
    done = False
    total_reward = 0

    print("Episode started")

    while not done:
        action = env.action_space.sample()  # Random action for now
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print("Episode finished.")
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    run_test()
