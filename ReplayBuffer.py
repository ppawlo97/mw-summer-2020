import random
from collections import deque
from typing import Any
from typing import Dict
from typing import List
from typing import Union


class ReplayBuffer:
    def __init__(self, max_size: int):
        """
        Agent's memory.

        Parameters
        ----------
        max_size: int
            Maximum memory size.

        """
        self.memory = deque(maxlen=max_size)


    def __len__(self):
        return len(self.memory)


    def store_step(self,
                   state: List[Any],
                   action: int,
                   reward: Union[float, int],
                   next_state: List[Any],
                   done: bool):
        """
        Stores environment's step in memory.
        
        Parameters
        ----------
        state: List[Any]
            Current state of the environment.
        action: int
            Integer representation of an action
            taken by the agent in the current state.
        reward: Union[float, int]
            Reward for the action taken in current state.
        next_state: List[Any]
            State of the environment after action was taken.
        done: bool
            Whether the episode is over.
        
        """
        step = dict(state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(done))
        self.memory.append(step)

    
    def sample(self, batch_size: int) -> Dict:
        """
        Samples `batch_size` historical steps from the memory.

        Parameters
        ----------
        batch_size: int
            Number of steps to sample.
        
        Returns
        -------
        Dict
        
        """
        batch = random.sample(self.memory, batch_size)
        batch = {key: [obs[key] for obs in batch]
                    for key in batch[0]}
        return batch