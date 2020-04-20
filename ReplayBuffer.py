import random
from collections import deque

# TODO: add docstring + type hints
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.memory = deque(maxlen=max_size)


    def __len__(self):
        return len(self.memory)


    def store_step(self, state, action, reward, next_state, done):
        step = dict(state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(done))
        self.memory.append(step)

    
    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        batch = {key: [obs[key] for obs in batch]
                    for key in batch[0]}
        return batch