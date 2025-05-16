import argparse
import yaml
import os
import torch
from datetime import datetime

from rl_training.agents import ppo_agent, sac_agent
from rl_training.utils.callbacks import TensorboardCallback
from rl_training.logger_callback import LoggerCallback
from stable_baselines3.common.callbacks import CallbackList

from jump_modular_env.logger import Logger


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_save_dirs(base_dir, algo, policy):
    save_dir = os.path.join(base_dir, algo.upper(), policy.lower())
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, "runs", current_time)

    model_dir = os.path.join(run_dir, "models")
    data_dir = os.path.join(run_dir, "data")
    tb_dir = os.path.join(run_dir, "tensorboard")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    return run_dir, model_dir, data_dir, tb_dir


def main(args):
    config = load_config(args.config)

    run_dir, model_dir, data_dir, tb_dir = make_save_dirs(
        args.logdir, args.algo, args.policy
    )

    if args.algo.lower() == "ppo":
        agent = ppo_agent.create_agent(
            policy_type=args.policy,
            config=config,
            tensorboard_log=tb_dir,
            n_envs=args.n_envs,
        )
    elif args.algo.lower() == "sac":
        agent = sac_agent.create_agent(
            policy_type=args.policy,
            config=config,
            tensorboard_log=tb_dir,
            n_envs=args.n_envs,
        )
    else:
        raise ValueError(f"Unknown algorithm {args.algo}")

    logger = Logger(
        log_interval=10,
        data_root=data_dir,
        model_save_path=model_dir,
        model_ref=agent,
    )

    callbacks = CallbackList(
        [
            TensorboardCallback(),
            LoggerCallback(logger, check_freq=5000),
        ]
    )

    agent.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # After training, save last model
    logger.save_last_model()
    print(f"Training finished. Models saved in: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument(
        "--policy", type=str, required=True, choices=["mlp", "cnn", "cnn3h"]
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file."
    )
    parser.add_argument(
        "--total-timesteps", type=int, required=True, help="Total training timesteps."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="models",
        help="Base directory to save models and logs.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of parallel environments."
    )

    args = parser.parse_args()

    main(args)
