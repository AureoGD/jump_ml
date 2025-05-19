from stable_baselines3.common.policies import MultiInputPolicy
from rl_training.policies.cnn3h_feature_extractor import CNN3HFeatureExtractor


class CNN3HPolicy(MultiInputPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CNN3HFeatureExtractor,
            features_extractor_kwargs=dict(
                past_channels=15,
                fut_channels=15,
                np_steps=10,
                nf_steps=5,
                nonpred_channels=11,
            ),
            **kwargs,
        )
