import gymnasium as gym
import yaml
import os
import sys  # For potentially modifying Python path

# --- Define your project's root directory or ensure custom_sac is in PYTHONPATH ---
# Example: If 'custom_sac' is a subdirectory of the directory containing 'run_trainer.py'
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # Or adjust as needed
# sys.path.insert(0, project_root) # Add project root to path
# Or, if 'custom_sac' is in the same directory as this script, no special path modification is needed.

from custom_sac.trainer import Trainer
# from custom_sac.agent import Agent # Agent is imported by Trainer

# Optional: If you define custom callbacks in a separate file or within trainer.py
# from custom_sac.trainer import BaseCallback
# class MyExampleCallback(BaseCallback):
#     def _on_episode_end(self):
#         if self.trainer.total_episodes_completed % 20 == 0: # Example: Log every 20 episodes
#             print(f"Callback: Episode {self.trainer.total_episodes_completed} ended. "
#                   f"Total timesteps: {self.trainer.total_timesteps}")
#         return True # Must return True to continue training


def load_config(config_path="config_custom_sac.yaml"):
    """Loads training configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration
    # You can make the config file path an argument, e.g., using argparse
    config_file = "config.yaml"
    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)

    # Create the training environment
    print(f"Creating training environment: {config['env_id']}")
    env = gym.make(config['env_id'])

    # Create the evaluation environment
    eval_env = None
    if config.get('eval_env_id'):  # Use .get for optional keys
        print(f"Creating evaluation environment: {config['eval_env_id']}")
        eval_env = gym.make(config['eval_env_id'])
    else:
        print(
            f"No separate evaluation environment specified. Using a new instance of training env for evaluation: {config['env_id']}"
        )
        # It's good practice to use a separate instance for evaluation, even if it's the same type
        eval_env = gym.make(config['env_id'])

    # Instantiate callbacks if needed
    # example_callback = MyExampleCallback(verbose=1)
    # callbacks_to_use = [example_callback]
    callbacks_to_use = None  # Set to None or an empty list if no callbacks initially

    # Initialize the Trainer with parameters from the config file
    print("Initializing Trainer...")
    max_grad_norm_from_config = config.get('max_grad_norm')
    if isinstance(max_grad_norm_from_config, (int, float)):
        max_grad_norm_for_trainer = float(max_grad_norm_from_config)
    else:
        max_grad_norm_for_trainer = None  # Default to None if not a valid number or missing

    trainer = Trainer(
        env=env,
        eval_env=eval_env,
        training_total_timesteps=config['training_total_timesteps'],
        max_steps_per_episode=config['max_steps_per_episode'],
        use_encoder=config.get('use_encoder', False),  # Default to False if not in YAML
        log_interval_timesteps=config.get('log_interval_timesteps', 1000),
        eval_frequency_timesteps=config.get('eval_frequency_timesteps', 5000),
        n_eval_episodes=config.get('n_eval_episodes', 5),
        log_root=config.get('log_root', "runs_custom_sac"),
        model_root=config.get('model_root', "models_custom_sac"),
        save_freq_episodes=config.get('save_freq_episodes', 100),
        callback=callbacks_to_use,

        # Agent specific HPs passed to Trainer, which then passes to Agent
        gamma=config.get('gamma', 0.99),
        tau=config.get('tau', 0.005),
        alpha=config.get('alpha', "auto"),
        critic_lr=config.get('critic_lr', 3e-4),
        actor_lr=config.get('actor_lr', 3e-4),
        replay_buffer_size=config.get('replay_buffer_size', 1000000),
        batch_size=config.get('batch_size', 256),
        learning_starts=config.get('learning_starts', 1000),
        gradient_steps=config.get('gradient_steps', 1),
        reward_scale=config.get('reward_scale', 1.0),
        max_grad_norm=max_grad_norm_for_trainer,
    )

    print("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing trainer and environments...")
        trainer.close()
        # env.close() and eval_env.close() are called by trainer.close()
        print("Cleanup complete. Training run finished.")


if __name__ == "__main__":
    main()
