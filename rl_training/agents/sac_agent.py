from stable_baselines3 import SAC
from rl_training.policies.mlp_policy import MLPSACPolicy
from rl_training.policies.cnn_sac_policy import CNNSACPolicy
from rl_training.policies.cnn3h_feature_extractor import CNN3HFeatureExtractor
from jump_modular_env.jump_env import JumperEnv
from stable_baselines3.common.vec_env import DummyVecEnv


def create_agent(policy_type="mlp", config=None, tensorboard_log=None, n_envs=1):
    """
    Create and return a SAC agent.

    Args:
        policy_type (str): "mlp", "cnn", or "cnn3h".
        config (dict): Hyperparameters and architecture settings.
        tensorboard_log (str): Path to TensorBoard log directory.
        n_envs (int): Number of parallel environments (SAC usually uses 1 env).
    Returns:
        SAC model.
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
            render=True,
            render_every=True,
            policy_type=policy_type,
            discrete_actions=False,
        )
    else:
        env = DummyVecEnv([make_env() for _ in range(n_envs)])

    # Select policy class
    if policy_type == "cnn":
        policy = CNNSACPolicy
    elif policy_type == "mlp":
        policy = MLPSACPolicy
    elif policy_type == "cnn3h":
        policy = "MultiInputPolicy"
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Set policy kwargs
    if policy_type == "cnn3h":
        policy_kwargs = dict(features_extractor_class=CNN3HFeatureExtractor)
    else:
        policy_kwargs = config.get("policy_kwargs", {})

    # Create SAC model
    model = SAC(
        policy,
        env,
        learning_rate=config["sac_hyperparams"]["learning_rate"],
        batch_size=config["sac_hyperparams"]["batch_size"],
        gamma=config["sac_hyperparams"]["gamma"],
        tau=config["sac_hyperparams"]["tau"],
        ent_coef=config["sac_hyperparams"].get("ent_coef", "auto"),
        train_freq=config["sac_hyperparams"].get("train_freq", (1, "step")),
        gradient_steps=config["sac_hyperparams"].get("gradient_steps", 1),
        learning_starts=config["sac_hyperparams"].get("learning_starts", 1000),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    return model
