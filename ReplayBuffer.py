import numpy as np
from typing import Dict


class ReplayBuffer:
    """
    Experience replay buffer for DQN-style agents.
    Stores fixed-size tuples of (obs, action, reward, next_obs, done).
    """

    def __init__(self, obs_dim: int, capacity: int, batch_size: int = 32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.size = 0
        self.ptr = 0

        # Make buffers for the (state, action, next_state, reward, done) 
        self.state_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity,), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity,), dtype=np.float32)
        self.done_buffer = np.zeros((capacity,), dtype=np.float32)

    def store(self, 
              state: np.ndarray, 
              action: float, 
              reward: float, 
              next_state: np.ndarray, 
              done: bool):
        self.state_buffer[self.ptr] = state  
        self.action_buffer[self.ptr] = action 
        self.reward_buffer[self.ptr] = reward 
        self.next_state_buffer[self.ptr] = next_state 
        self.done_buffer[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self):
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        return {
            "obs": self.obs_buf[idxs],
            "acts": self.acts_buf[idxs],
            "rews": self.rews_buf[idxs],
            "next_obs": self.next_obs_buf[idxs],
            "done": self.done_buf[idxs],
        }

    def __len__(self) -> int:
        return self.size
