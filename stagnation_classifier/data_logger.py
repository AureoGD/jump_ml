import os
import pickle
import numpy as np

from jump_modular_env.jump_env import JumperEnv


class DataLogger:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_episode(self, env, ep_number):
        ml = env.get_wrapper_attr("ml_wrapper")

        base_past = np.stack(list(ml.base_past), axis=1)  # (10, NP)
        joint_past = np.stack(list(ml.joint_past), axis=1)  # (12, NP)
        comp_past = np.stack(list(ml.comp_states), axis=1)  # (5, NP)

        base_future = ml.pred_st[0:10, :]  # (10, NF)
        joint_future = ml.pred_st[10:, :]  # (12, NF)

        data = {
            "base_past": base_past,
            "joint_past": joint_past,
            "comp_past": comp_past,
            "base_future": base_future,
            "joint_future": joint_future,
        }

        file_path = os.path.join(self.save_dir, f"episode_{ep_number:03d}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ Saved {file_path}")
