from collections import deque 
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        assert batch_size <= len(self), "Batch size cannot be greater than the size of the deque"
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)