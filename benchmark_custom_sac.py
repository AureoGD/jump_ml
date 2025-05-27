import gymnasium as gym
from custom_sac.trainer import Trainer

# Create the environment externally
env = gym.make("Pendulum-v1")

# Pass it to the trainer
trainer = Trainer(
    env=env,
    n_episodes=1000,
    max_steps=1000,
    use_encoder=False,
)

trainer.train()
trainer.close()
env.close()
