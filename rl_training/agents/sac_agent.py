# # rl_training/agents/sac_agent.py

# from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import DummyVecEnv

# from rl_training.policies.mlp_policy import MLPSACPolicy
# from rl_training.policies.cnn_sac_policy import CNNSACPolicy
# from rl_training.policies.cnn3h_feature_extractor import CNN3HFeatureExtractor
# from rl_training.policies.multi_head_extractor import MultiHeadFeatureExtractor
# from rl_training.policies.multi_output_sac_policy import MultiOutputSACPolicy

# from rl_training.agents.multi_output_sac import MultiOutputSAC
# from rl_training.utils.custom_replay_buffer import MultiOutputReplayBuffer

# from jump_modular_env.jump_env import JumperEnv

# from stable_baselines3.common.buffers import DictReplayBuffer

# def create_agent(policy_type="mlp", config=None, tensorboard_log=None, n_envs=1):
#     """
#     Generic SAC agent factory.

#     Args:
#         policy_type (str): ["mlp", "cnn", "cnn3h", "multihead"]
#         config (dict): YAML config dictionary.
#         tensorboard_log (str): TensorBoard log path.
#         n_envs (int): Number of parallel envs (suggested 1 for SAC).

#     Returns:
#         SAC or MultiOutputSAC instance.
#     """

#     def make_env():

#         def _init():
#             env = JumperEnv(
#                 render=False,
#                 policy_type=policy_type,
#                 discrete_actions=False,
#                 disturb=False,
#             )
#             return env

#         return _init

#     if n_envs == 1:
#         env = JumperEnv(
#             render=True,
#             render_every=True,
#             policy_type=policy_type,
#             discrete_actions=False,
#         )
#     else:
#         env = DummyVecEnv([make_env() for _ in range(n_envs)])

#     # -----------------------------
#     # Policy and SAC class selection
#     # -----------------------------
#     if policy_type == "mlp":
#         policy = MLPSACPolicy
#         sac_class = SAC
#         replay_buffer_class = None

#     elif policy_type == "cnn":
#         policy = CNNSACPolicy
#         sac_class = SAC
#         replay_buffer_class = None

#     elif policy_type == "cnn3h":
#         policy = "MultiInputPolicy"
#         sac_class = SAC
#         replay_buffer_class = None

#     elif policy_type == "multihead":
#         policy = MultiOutputSACPolicy
#         sac_class = MultiOutputSAC
#         replay_buffer_class = MultiOutputReplayBuffer  # ✔️ For custom replay

#     else:
#         raise ValueError(f"Unknown policy type: {policy_type}")

#     # -----------------------------
#     # Policy kwargs
#     # -----------------------------
#     if policy_type == "cnn3h":
#         policy_kwargs = dict(features_extractor_class=CNN3HFeatureExtractor)

#     elif policy_type == "multihead":
#         policy_kwargs = dict(
#             features_extractor_class=MultiHeadFeatureExtractor,
#             features_extractor_kwargs={},
#         )

#     else:
#         policy_kwargs = config.get("policy_kwargs", {})

#     # -----------------------------
#     # Instantiate SAC
#     # -----------------------------
#     model = sac_class(
#         policy,
#         env,
#         learning_rate=config["sac_hyperparams"]["learning_rate"],
#         batch_size=config["sac_hyperparams"]["batch_size"],
#         gamma=config["sac_hyperparams"]["gamma"],
#         tau=config["sac_hyperparams"]["tau"],
#         ent_coef=config["sac_hyperparams"].get("ent_coef", "auto"),
#         train_freq=config["sac_hyperparams"].get("train_freq", 1),
#         gradient_steps=config["sac_hyperparams"].get("gradient_steps", 1),
#         learning_starts=config["sac_hyperparams"].get("learning_starts", 1000),
#         policy_kwargs=policy_kwargs,
#         tensorboard_log=tensorboard_log,
#         replay_buffer_class=replay_buffer_class,
#         verbose=1,
#     )

#     return model
# rl_training/agents/sac_agent.py

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from jump_modular_env.jump_env import JumperEnv

from rl_training.policies.multi_head_extractor import MultiHeadFeatureExtractor


def create_agent(policy_type="multihead", config=None, tensorboard_log=None, n_envs=1):
    """
    Create SAC agent with multi-head feature extractor.

    Args:
        policy_type (str): Only "multihead" supported for now.
        config (dict): Loaded YAML config.
        tensorboard_log (str): TensorBoard log directory.
        n_envs (int): Number of parallel envs.

    Returns:
        SAC model.
    """

    def make_env():
        return lambda: JumperEnv(
            policy_type=policy_type,
            discrete_actions=False,
        )

    if n_envs == 1:
        env = JumperEnv(policy_type=policy_type, discrete_actions=False)
    else:
        env = DummyVecEnv([make_env() for _ in range(n_envs)])

    if policy_type == "multihead":
        policy = "MultiInputPolicy"  # ✅ This is mandatory for Gym Dict observations
        policy_kwargs = dict(features_extractor_class=MultiHeadFeatureExtractor)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

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
