"""
Deep Q learning agent, that learns only through the interaction with
environment. Inherently designed only for discrete action spaces.

References
----------
    [1] https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning

"""
import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import tensorflow as tf
import yaml
from absl import logging
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

from agent.MLP import MLP
from agent.ReplayBuffer import ReplayBuffer 


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
        self._num_actions = num_actions
        self._mlp_hidden_units = mlp_hidden_units
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._initial_eps = epsilon # for saving
        self._eps = {"epsilon": epsilon,
                     "eps_decrement": eps_decrement,
                     "eps_minimum": eps_minimum}
        self._memory_size = memory_size
        self._min_memory = min_memory
        self._seed = seed
        
        random.seed(self._seed)
        tf.random.set_seed(self._seed)

        self._memory = ReplayBuffer(self._memory_size)
        self._net = MLP(self._num_actions, self._mlp_hidden_units)
        self._optimizer = Adam(self._learning_rate)

        self._ckpt = tf.train.Checkpoint(optimizer=self._optimizer,
                                         model=self._net)


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
            action = random.randrange(self._num_actions)
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
                             depth=self._num_actions)
        
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

    
    def save(self, save_dir: str) -> str:
        """
        Saves the agent along with model checkpoint and
        configuration parameters.

        Parameters
        ----------
        save_dir: str
            Directory where to save the agent.

        Returns
        -------
        Path to the folder with saved model.

        """
        save_dir = Path(save_dir)
        if not save_dir.exists() or not save_dir.is_dir():
            save_dir.mkdir(parents=True)
        else:
            logging.warning(f"Overwriting existing directory {save_dir} !")

        model_dir = save_dir.joinpath("dqn")
        if not model_dir.exists() or not model_dir.is_dir():
            model_dir.mkdir()
        else:
            logging.warning(f"Overwriting existing directory {model_dir} !")

        params_path = model_dir.joinpath("params.yml")
        logging.info(f"Saving DQN's params to {params_path}")
        self._to_yaml(params_path)
        

        model_path = model_dir.joinpath("mlp")
        logging.info(f"Saving Q Net checkpoint to {model_path}")
        self._ckpt.save(model_path)

        return model_dir.as_posix()


    def _to_yaml(self, save_path: Path):
        """
        Dumps DQN parameters to YAML.
        
        Parameters
        ----------
        save_path: Path
            Where to save the YAML file.
        
        """
        params = self.get_config()
        
        with open(save_path, "w") as stream:
            yaml.dump(params, stream)


    @classmethod
    def load(cls, model_dir: str) -> "DQN":
        """
        Loads trained DQN.

        Parameters
        ----------
        model_dir: str
            Directory with DQN's configuration
            and Q Network checkpoint.

        Returns
        -------
        DQN

        """
        model_dir = Path(model_dir)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"{model_dir} doesn't exist or isn't a directory!")

        params_path = model_dir.joinpath("params.yml")
        with open(params_path, "r") as stream:
            params = yaml.full_load(stream)

        latest_ckpt = tf.train.latest_checkpoint(model_dir)

        logging.info(f"Loading DQN from {model_dir}.")
        dqn = cls(**params)
        dqn._ckpt.restore(latest_ckpt).expect_partial()

        return dqn

    
    def get_config(self) -> Dict[str, Any]:
        """Gets DQN's config."""
        config = {
            "num_actions": self._num_actions,
            "mlp_hidden_units": self._mlp_hidden_units,
            "learning_rate": self._learning_rate,
            "discount_factor": self._discount_factor,
            "memory_size": self._memory_size,
            "min_memory": self._min_memory,
            "seed": self._seed
        }
        config.update(self._eps)
        config["epsilon"] = self._initial_eps
        return config