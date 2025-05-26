# custom_sac/replay_buffer.py

import numpy as np


class ReplayBuffer:

    def __init__(self, max_size, obs_space_dict, n_actions):
        self.max_size = max_size
        self.mem_cntr = 0

        # Initialize observation buffers for each dict key
        self.state_memory = {
            key: np.zeros((max_size, *space), dtype=np.float32)
            for key, space in obs_space_dict.items()
        }
        self.new_state_memory = {
            key: np.zeros((max_size, *space), dtype=np.float32)
            for key, space in obs_space_dict.items()
        }

        # Action, reward, terminal
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.max_size

        for key in self.state_memory.keys():
            self.state_memory[key][index] = state[key]
            self.new_state_memory[key][index] = state_[key]

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = {key: self.state_memory[key][batch] for key in self.state_memory.keys()}
        new_states = {key: self.new_state_memory[key][batch] for key in self.new_state_memory.keys()}

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

    def size(self):
        return min(self.mem_cntr, self.max_size)
