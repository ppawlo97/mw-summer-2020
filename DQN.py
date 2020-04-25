import random
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

from MLP import MLP
from ReplayBuffer import ReplayBuffer 

# TODO: add saving and loading from YAML
class DQN:
    def __init__(self,
                 num_actions: int,
                 mlp_hidden_units: Tuple[int] = (256, 256),
                 learning_rate: float = 0.00005,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.8,
                 eps_decrement: float = 0.99,
                 eps_minimum: float = 0.01,
                 memory_size: int = 20_000,
                 min_memory: int = 3000,
                 seed: int = 42):
        """
        Deep Q Learning Agent
        with Multilayer Perceptron Q Network.

        Parameters
        ----------
        num_actions: int
            Number of actions that the agent
            can take in the environment.
        mlp_hidden_units: Tuple[int]
            Number of units for every hidden layer
            of the MLP Q Network.
        learning_rate: float
            Optimization step size.
        discount_factor: float
            Factor that determines how much agent
            takes future rewards into account.
        epsilon: float
            Initial probability of taking a random action.
            A.k.a. `epsilon` from epsilon-greedy exploration strategy
        eps_decrement: float
            Decrement factor of epsilon parameter.
        eps_minimum: float
            Minimum epsilon.
            Epsilon will not be reduced below this value.
        memory_size: int
            Maximum number of past steps stored in the ReplayBuffer.
        min_memory: int
            Minimum number of past steps in the ReplayBuffer
            before agent starts learning.
        seed: int
            Random state for RNG.

        """
        random.seed(seed)
        tf.random.set_seed(seed)

        self._memory = ReplayBuffer(memory_size)
        self._net = MLP(num_actions, mlp_hidden_units)
        self._optimizer = Adam(learning_rate)

        self._min_memory = min_memory
        self._discount_factor = discount_factor
        self._action_space = list(range(num_actions))
        self._eps = {"epsilon": epsilon,
                     "eps_decrement": eps_decrement,
                     "eps_minimum": eps_minimum}


    def extend_memory(self,
                      state: List[Any],
                      action: int,
                      reward: Union[float, int],
                      next_state: List[Any],
                      done: bool):
        """
        Extends agent's memory with additional
        environment step.
        
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
        self._memory.store_step(state=state,
                                action=action,
                                reward=reward, 
                                next_state=next_state,
                                done= done)

    
    def predict(self, state: List[Union[float, int]]) -> int:
        """
        Determines which action should be taken
        given current state of the environment
        
        Notes
        -----
        Sometimes, the agent takes random action with
        probability epsilon.

        Parameters
        ----------
        state: List[Union[float, int]]
            Current state of the environment represented
            in numerical form.

        Returns
        -------
        int
            Selected action
        
        """
        if random.random() < self._eps["epsilon"]:
            action = random.choice(self._action_space)
        else:
            state = tf.expand_dims(state, axis=0)
            action = tf.argmax(self._net(state),
                               axis=-1)[0].numpy()
        
        return action


    def fit(self, batch_size: int):
        """
        Trains the agent.

        Parameters
        ----------
        batch_size: int
            Number of memories to fit the Q Network on.
        
        """
        if max(self._min_memory, batch_size) > len(self._memory):
            return

        batch = self._memory.sample(batch_size)

        states = tf.stack(batch["state"], axis=0)
        next_states = tf.stack(batch["next_state"], axis=0)
        actions = tf.one_hot(indices=batch["action"],
                             depth=len(self._action_space))
        
        q_values_next = tf.reduce_max(self._net(next_states), axis=-1)
        q_target = (tf.convert_to_tensor(batch["reward"]) +
                    self._discount_factor * 
                    q_values_next * 
                    (1 - tf.convert_to_tensor(batch["done"])))
        
        with tf.GradientTape() as tape:
            q_values = self._net(states)
            prediction = tf.reduce_sum(q_values * actions, axis=1)
            loss = mean_squared_error(q_target, prediction)
        weights = self._net.trainable_variables
        gradients = tape.gradient(loss, weights)
        self._optimizer.apply_gradients(zip(gradients, weights))

        if self._eps["eps_minimum"] < self._eps["epsilon"]:
            self._eps["epsilon"] *= self._eps["eps_decrement"]