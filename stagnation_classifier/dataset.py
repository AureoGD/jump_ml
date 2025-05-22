import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd


class StagnationDataset(Dataset):

    def __init__(self, file_list, augment=False):
        """
        Dataset for stagnation regression using multihead input.

        Args:
            file_list (list): List of .pkl file paths.
            augment (bool): If True, apply Gaussian noise to the input data.
        """
        self.samples = []
        self.labels = []
        self.augment = augment

        for file_path in file_list:
            with open(file_path, "rb") as f:
                episode = pickle.load(f)

            for step in episode:
                base = step["base_past"]  # (6, T) including CoM
                joint = step["joint_past"]  # (12, T)
                comp = step["comp_past"]  # (5, T)
                label = step["label"]  # float label between 0 and 1

                # Check shapes
                assert base.shape[0] == 10, f"base_past has {base.shape[0]} channels, expected 10"
                assert joint.shape[0] == 12, f"joint_past has {joint.shape[0]} channels, expected 12"
                assert comp.shape[0] == 5, f"comp_past has {comp.shape[0]} channels, expected 5"

                if self.augment:
                    base, joint, comp = self.apply_augmentation(base, joint, comp)

                self.samples.append((base, joint, comp))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base, joint, comp = self.samples[idx]
        label = self.labels[idx]
        return {
            "base_past": torch.tensor(base, dtype=torch.float32),
            "joint_past": torch.tensor(joint, dtype=torch.float32),
            "comp_past": torch.tensor(comp, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def apply_augmentation(self, base, joint, comp):
        noise_base = base + np.random.normal(0, 0.001, size=base.shape)
        noise_joint = joint + np.random.normal(0, 0.001, size=joint.shape)
        noise_comp = comp + np.random.normal(0, 0.001, size=comp.shape)
        return noise_base, noise_joint, noise_comp


def load_dataset(manual_dir, test_size=0.2, seed=42, augment=False):
    files = sorted([os.path.join(manual_dir, f) for f in os.listdir(manual_dir) if f.endswith(".pkl")])

    train_files, test_files = train_test_split(files, test_size=test_size, random_state=seed)

    train_dataset = StagnationDataset(train_files, augment=augment)
    test_dataset = StagnationDataset(test_files, augment=False)

    return train_dataset, test_dataset


def summarize_dataset(dataset, name):
    labels = np.array([sample["label"].item() for sample in dataset])
    return {
        "Name": name,
        "Total": len(labels),
        "Mean Label": labels.mean(),
        "Min": labels.min(),
        "Max": labels.max(),
    }


if __name__ == "__main__":
    manual_dir = "stagnation_classifier/labeled_episodes_manual"
    train_dataset, test_dataset = load_dataset(manual_dir)

    summary = [
        summarize_dataset(train_dataset, "Train"),
        summarize_dataset(test_dataset, "Test"),
    ]

    df = pd.DataFrame(summary)
    print(df)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.bar(df["Name"], df["Mean Label"], color="purple", alpha=0.7)
    plt.title("Average Stagnation Score per Dataset")
    plt.ylabel("Average Label Value (0-1)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
