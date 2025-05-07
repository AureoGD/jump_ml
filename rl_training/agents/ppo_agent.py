# rl_training/agents/ppo_agent.py

from stable_baselines3 import PPO
from rl_training.policies.cnn_policy import CNNPPOPolicy
from rl_training.policies.mlp_policy import MLPPPOPolicy
from jump_modular_env.jump_env import JumperEnv
from stable_baselines3.common.vec_env import DummyVecEnv


def create_agent(policy_type="mlp", config=None, tensorboard_log=None, n_envs=1):
    """
    Create and return a PPO agent.

    Args:
        policy_type (str): "mlp" or "cnn".
        config (dict): Hyperparameters and architecture settings.
        tensorboard_log (str): Path to TensorBoard log directory.
        n_envs (int): Number of parallel environments.
    Returns:
        PPO model.
    """

    def make_env():
        def _init():
            env = JumperEnv(render=False, policy_type=policy_type)
            return env

        return _init

    if n_envs == 1:
        env = JumperEnv(render=False, policy_type=policy_type)
    else:
        env = DummyVecEnv([make_env() for _ in range(n_envs)])

    # Select policy class
    if policy_type == "cnn":
        policy = CNNPPOPolicy
    elif policy_type == "mlp":
        policy = MLPPPOPolicy
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Create PPO model
    model = PPO(
        policy,
        env,
        learning_rate=config["ppo_hyperparams"]["learning_rate"],
        n_steps=config["ppo_hyperparams"]["n_steps"],
        batch_size=config["ppo_hyperparams"]["batch_size"],
        gamma=config["ppo_hyperparams"]["gamma"],
        gae_lambda=config["ppo_hyperparams"]["gae_lambda"],
        clip_range=config["ppo_hyperparams"]["clip_range"],
        ent_coef=config["ppo_hyperparams"].get("ent_coef", 0.0),
        tensorboard_log=tensorboard_log,
        verbose=1,
        policy_kwargs=config.get("policy_kwargs", {}),
        n_epochs=config["ppo_hyperparams"].get("n_epochs", 10),
    )

    return model
