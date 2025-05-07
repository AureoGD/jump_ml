import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt


class StagnationDataset(Dataset):
    def __init__(
        self,
        file_list,
        selected_predictable_idx,
        selected_nonpredictable_idx,
        augment=False,
    ):
        self.samples = []
        self.labels = []
        self.augment = augment

        for file_path in file_list:
            with open(file_path, "rb") as f:
                episode = pickle.load(f)

            for step in episode:
                pred = step["predictable"][
                    selected_predictable_idx, :, 0
                ]  # shape: (num_selected, time)
                nonp = step["nonpredictable"][
                    selected_nonpredictable_idx, :, 0
                ]  # shape: (num_selected, time)
                label = step["label"]

                if self.augment:
                    pred, nonp = self.apply_augmentation(pred, nonp)

                self.samples.append((pred, nonp))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pred, nonp = self.samples[idx]
        label = self.labels[idx]
        return {
            "predictable": torch.tensor(pred, dtype=torch.float32),
            "nonpredictable": torch.tensor(nonp, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def apply_augmentation(self, pred, nonp):
        noise_pred = pred + np.random.normal(0, 0.001, size=pred.shape)
        noise_nonp = nonp + np.random.normal(0, 0.001, size=nonp.shape)
        return noise_pred, noise_nonp


def load_dataset(manual_dir, test_size=0.2, seed=42, augment=False):
    manual_files = sorted(
        [
            os.path.join(manual_dir, f)
            for f in os.listdir(manual_dir)
            if f.endswith(".pkl")
        ]
    )

    all_files = manual_files
    train_files, test_files = train_test_split(
        all_files, test_size=test_size, random_state=seed
    )

    # Indices of selected features
    pred_indices = [4, 5]  # theta_y, dtheta_y
    nonp_indices = [7, 8, 9, 10, 3]  # vel_x, vel_z, pos_x, pos_z, controller_mode

    train_dataset = StagnationDataset(
        train_files, pred_indices, nonp_indices, augment=augment
    )
    test_dataset = StagnationDataset(
        test_files, pred_indices, nonp_indices, augment=False
    )

    return train_dataset, test_dataset


# import jumpstat  # Show the datasets to the user


def summarize_dataset(dataset, name):
    labels = [sample["label"].item() for sample in dataset]
    total = len(labels)
    pos = sum(labels)
    neg = total - pos
    return {"Name": name, "Total": total, "Positive": pos, "Negative": neg}


# Load and summarize for display
manual_dir = "stagnation_classifier/labeled_episodes_manual"
train_dataset, test_dataset = load_dataset(manual_dir)

summary = [
    summarize_dataset(train_dataset, "Train"),
    summarize_dataset(test_dataset, "Test"),
]


# Summarized data
summary = [
    {"Name": "Train", "Total": 4825, "Positive": 1364, "Negative": 3461},
    {"Name": "Test", "Total": 1207, "Positive": 321, "Negative": 886},
]

# Create DataFrame
df = pd.DataFrame(summary)

# # Plot table
# fig, ax = plt.subplots(figsize=(6, 2))
# ax.axis("off")
# table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1, 2)

# plt.tight_layout()
# plt.show()
