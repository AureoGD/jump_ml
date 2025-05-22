import os
import pickle
import numpy as np

manual_dir = "stagnation_classifier/labeled_episodes_manual"
file_list = [os.path.join(manual_dir, f) for f in os.listdir(manual_dir) if f.endswith(".pkl")]

sample_file = file_list[0]
with open(sample_file, "rb") as f:
    episode = pickle.load(f)

step = episode[0]

print(f"base_past shape: {step['base_past'].shape}")
print(f"joint_past shape: {step['joint_past'].shape}")
print(f"comp_past shape: {step['comp_past'].shape}")
