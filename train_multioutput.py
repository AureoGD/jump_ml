# rl_training/agents/sac_agent.py

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_training.policies.multi_head_extractor import MultiHeadFeatureExtractor
from jump_modular_env.jump_env import JumperEnv


def create_agent(policy_type="mlp", config=None, tensorboard_log=None, n_envs=1):
    """
    SAC agent factory using vanilla SAC.
    Supports 'multihead' with MultiInputPolicy + MultiHeadFeatureExtractor.
    """

    def make_env():

        def _init():
            env = JumperEnv(
                render=False,
                policy_type=policy_type,
                discrete_actions=False,
                disturb=False,
            )
            return env

        return _init

    if n_envs == 1:
        env = JumperEnv(
            render=False,
            policy_type=policy_type,
            discrete_actions=False,
            disturb=False,
        )
    else:
        env = DummyVecEnv([make_env() for _ in range(n_envs)])

    if policy_type == "multihead":
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=MultiHeadFeatureExtractor,
            features_extractor_kwargs={},  # You can add if needed
        )
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")

    model = SAC(
        policy,
        env,
        learning_rate=config["sac_hyperparams"]["learning_rate"],
        batch_size=config["sac_hyperparams"]["batch_size"],
        gamma=config["sac_hyperparams"]["gamma"],
        tau=config["sac_hyperparams"]["tau"],
        ent_coef=config["sac_hyperparams"].get("ent_coef", "auto"),
        train_freq=config["sac_hyperparams"].get("train_freq", 1),
        gradient_steps=config["sac_hyperparams"].get("gradient_steps", 1),
        learning_starts=config["sac_hyperparams"].get("learning_starts", 1000),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    return model
