# stagnation_classifier/cnn_stagnation_wrapper.py

import torch
import numpy as np
from .models import CNNStagnationClassifier


class StagnationClassifier:

    def __init__(self, model_path, device="cpu"):
        """
        Wrapper for the stagnation regression model.

        Args:
            model_path (str): Path to the saved model (.pth with metadata)
            device (str): "cpu" or "cuda"
        """
        self.device = torch.device(device)
        self.seq_len = 30  # Should match what was used in training

        # Expected channels per input group
        self.base_channels = 10  # r, dr, th, dth, b, db
        self.joint_channels = 12  # q, dq, qr, tau
        self.comp_channels = 5  # stagnation, phase, transition_history, foot_contact, success

        self.input_channels = self.base_channels + self.joint_channels + self.comp_channels

        # Initialize model
        self.model = CNNStagnationClassifier(in_channels=self.input_channels, seq_len=self.seq_len).to(self.device)

        # Load model with metadata check
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata", {})
            saved_channels = metadata.get("input_channels", None)

            if saved_channels is not None and saved_channels != self.input_channels:
                raise ValueError(f"Input channel mismatch: Model expects {self.input_channels}, "
                                 f"but checkpoint was trained with {saved_channels} channels.")
        else:
            # Handle old model format without metadata
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, base_past_deque, joint_past_deque, comp_past_deque):
        """
        Predict stagnation score based on observation window.

        Args:
            base_past_deque: deque of (base) arrays
            joint_past_deque: deque of (joint) arrays
            comp_past_deque: deque of (comp) arrays

        Returns:
            float: stagnation score between 0 and 1
        """
        # Stack observations along time axis
        base_np = np.stack(list(base_past_deque), axis=1)  # Shape (C, T)
        joint_np = np.stack(list(joint_past_deque), axis=1)  # Shape (C, T)
        comp_np = np.stack(list(comp_past_deque), axis=1)  # Shape (C, T)

        # Validate shape
        assert base_np.shape[1] == self.seq_len, (
            f"Expected sequence length {self.seq_len}, but got {base_np.shape[1]}")
        assert joint_np.shape[1] == self.seq_len, (
            f"Expected sequence length {self.seq_len}, but got {joint_np.shape[1]}")
        assert comp_np.shape[1] == self.seq_len, (
            f"Expected sequence length {self.seq_len}, but got {comp_np.shape[1]}")

        # Combine channels
        combined = np.concatenate([base_np, joint_np, comp_np], axis=0)  # (C_total, T)

        assert combined.shape[0] == self.input_channels, (f"Input channel mismatch! Expected {self.input_channels}, "
                                                          f"but got {combined.shape[0]}")

        # Convert to tensor
        input_tensor = (
            torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # Add batch dimension: (1, C, T)
            .to(self.device))

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)  # Shape (1, 1)
            prediction = output.item()

        return prediction
