import torch
import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy


class MultiOutputSACPolicy(SACPolicy):
    """
    SAC Policy with auxiliary outputs:
    - success ∈ [0, 1]
    - stagnation ∈ [0, 1]
    - phase logits ∈ ℝ⁴ (classification)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy initialization placeholders
        self.success_head = None
        self.stagnation_head = None
        self.phase_head = None

    def _ensure_aux_heads(self):
        if self.success_head is not None:
            return

        feat_dim = self.features_extractor.features_dim

        self.success_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.stagnation_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.phase_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # logits
        )

    def predict_aux_outputs(self, obs):
        self._ensure_aux_heads()

        features = self.extract_features(obs)

        return {
            "success": self.success_head(features),
            "stagnation": self.stagnation_head(features),
            "phase": self.phase_head(features),
        }
