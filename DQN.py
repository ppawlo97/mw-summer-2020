import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from MLP import MLP
from ReplayBuffer import ReplayBuffer 

# TODO: add type hints + docstrings
class DQN:
    def __init__(self,
                 num_actions,
                 learning_rate = 0.05,
                 discount_factor = 0.99,
                 epsilon = 1,
                 epsilon_dec = 0.99,
                 epsilon_end = 0.01,
                 memory_size = 1_000_000):
        
        self.memory = ReplayBuffer(memory_size)
        self.net = MLP(num_actions)
        self.net.compile(optimizer=Adam(learning_rate), loss=MeanSquaredError())
    
        self.discount_factor = discount_factor
        self.action_space = list(range(num_actions))
        self.eps = {"epsilon": epsilon,
                    "epsilon_dec": epsilon_dec,
                    "epsilon_end": epsilon_end}


    def extend_memory(self, state, action, reward, next_state, done):
        self.memory.store_step(state, action, reward, next_state, done)

    
    def predict(self, state):
        if random.random() < self.eps["epsilon"]:
            action = random.choice(self.action_space)
        else:
            state = tf.expand_dims(state, axis=0)
            action = tf.argmax(self.net(state),
                               axis=-1)[0].numpy()
        
        return action


    def fit(self, batch_size):
        if batch_size > len(self.memory):
            return

        batch = self.memory.sample(batch_size)

        states = tf.stack(batch["state"], axis=0)
        next_states = tf.stack(batch["next_state"], axis=0)
        actions = tf.one_hot(indices=batch["action"],
                             depth=len(self.action_space))
        actions = tf.cast(actions, dtype=tf.bool)

        q_values = self.net(states)
        q_values_next = tf.reduce_max(self.net(next_states), axis=-1)

        discounted_rewards = (tf.convert_to_tensor(batch["reward"]) +
                              self.discount_factor * 
                              q_values_next * 
                              (1 - tf.convert_to_tensor(batch["done"])))
        discounted_rewards = tf.repeat(tf.expand_dims(discounted_rewards, -1),
                                       repeats=2,
                                       axis=-1)
        
        q_target = tf.where(actions,
                            discounted_rewards,
                            q_values)
        
        self.net.fit(states, q_target, verbose=0)

        if self.eps["epsilon_end"] > self.eps["epsilon"]:
            self.eps["epsilon"] *= self.eps["epsilon_dec"]