import os
import pickle
import numpy as np
import pybullet as p
from tqdm import trange
from jump_modular_env.jump_env import JumperEnv

SAVE_DIR = "stagnation_classifier/raw_episodes"
VIDEO_DIR = "stagnation_classifier/videos"
NUM_EPISODES = 20

MAX_STEPS = 1000

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


def extract_observation_window(env):
    """
    Extracts stacked state history from the ml_wrapper.
    Assumes ml_wrapper.predictable_history is a deque of 2D arrays (features per timestep).
    """
    predictable = np.stack(list(env.ml_wrapper.predictable_history), axis=1)
    nonpredictable = np.stack(list(env.ml_wrapper.nonpredictable_history), axis=1)
    return {"predictable": predictable, "nonpredictable": nonpredictable}


def run_episode(env, episode_idx, transition_step, mode_1, mode_2):
    print(f"\n--- Episode {episode_idx + 1} ---")
    obs, _ = env.reset()
    episode_log = []

    # Start video recording
    video_path = os.path.join(SAVE_DIR, f"episode_{episode_idx:03d}.mp4")
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)

    for step in range(MAX_STEPS):
        # Mode policy
        if step < transition_step:
            action = [mode_1]
        else:
            action = [mode_2]

        obs, reward, terminated, truncated, info = env.step(action)

        window = extract_observation_window(env)
        episode_log.append(window)

        if terminated or truncated:
            break

    # Stop recording
    p.stopStateLogging(log_id)
    print(f"🎥 Saved video to {video_path}")

    return episode_log, step


def main():
    env = JumperEnv(render=True, disturb=False, policy_type="cnn")

    ep = 19
    while ep < NUM_EPISODES + 19:
        print(f"\n--- Episode {ep + 1}/{NUM_EPISODES + 19} ---")

        # User-defined transition and modes
        transition_step = int(input("Transition step (e.g., 750): ").strip())
        mode_1 = float(input("First mode (before transition): ").strip())
        mode_2 = float(input("Second mode (after transition): ").strip())

        episode_log, last_step = run_episode(env, ep, transition_step, mode_1, mode_2)

        print(f"Episode ran for {last_step} steps.")
        user_input = input(f"Save episode {ep + 1}? (y/n): ").strip().lower()
        if user_input == "y":
            file_path = os.path.join(SAVE_DIR, f"episode_{ep:03d}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "data": episode_log,
                        "transition_step": transition_step,
                        "mode_1": mode_1,
                        "mode_2": mode_2,
                    },
                    f,
                )
            print(f"✅ Saved to {file_path}")
            ep += 1
        else:
            print("❌ Skipped saving. Repeating episode...")


if __name__ == "__main__":
    main()
