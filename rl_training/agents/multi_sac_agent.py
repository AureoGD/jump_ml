from stable_baselines3 import SAC
from rl_training.policies.multi_output_sac_policy import MultiOutputSACPolicy
from rl_training.policies.multi_head_extractor import MultiHeadFeatureExtractor
from rl_training.utils.custom_replay_buffer import MultiOutputReplayBuffer
from rl_training.agents.multi_output_sac import MultiOutputSAC
from jump_modular_env.jump_env import JumperEnv


def create_agent(policy_type, config, tensorboard_log=None, n_envs=1):
    env = JumperEnv(policy_type=policy_type, discrete_actions=False)

    if policy_type == "multihead":
        sac_class = MultiOutputSAC
        policy_class = MultiOutputSACPolicy
        replay_buffer_class = MultiOutputReplayBuffer
        policy_kwargs = dict(features_extractor_class=MultiHeadFeatureExtractor)
    else:
        raise NotImplementedError()

    model = sac_class(
        policy_class,
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
        replay_buffer_class=replay_buffer_class,
        verbose=1,
    )

    return model
