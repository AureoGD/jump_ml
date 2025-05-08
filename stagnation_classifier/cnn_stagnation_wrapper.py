# stagnation_classifier/cnn_stagnation_wrapper.py

import torch
import numpy as np
from .models import CNNStagnationClassifier


class StagnationClassifier:
    def __init__(self, model_path, device="cpu"):
        # Indices of features to extract
        self.pred_indices = [4, 5]  # theta_y, dtheta_y
        self.nonp_indices = [7, 8, 9, 10, 3]  # vel_x, vel_z, pos_x, pos_z, mode

        self.device = torch.device(device)
        self.seq_len = 30  # should match training

        self.total_channels = len(self.pred_indices) + len(self.nonp_indices)

        self.model = CNNStagnationClassifier(
            in_channels=self.total_channels, seq_len=self.seq_len
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, predictable_deque, nonpredictable_deque):
        # Extract and stack data
        pred_np = np.stack(list(predictable_deque), axis=1)[
            self.pred_indices
        ]  # (2, 30)
        nonp_np = np.stack(list(nonpredictable_deque), axis=1)[
            self.nonp_indices
        ]  # (5, 30)

        combined = np.concatenate([pred_np, nonp_np], axis=0)  # (7, 30)

        # Convert to tensor
        input_tensor = (
            torch.tensor(combined, dtype=torch.float32)
            .unsqueeze(0)  # batch dim
            .squeeze(-1)  # remove final channel dim if present
            .to(self.device)
        )  # Shape: (1, in_channels, seq_len)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction
