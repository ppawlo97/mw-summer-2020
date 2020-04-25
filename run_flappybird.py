import os

import numpy as np
from absl import app
from absl import flags
from absl import logging
from ple import PLE
from ple.games.flappybird import FlappyBird

from DQN import DQN


FLAGS = flags.FLAGS

flags.DEFINE_bool(name="display_screen",
                  default=True,
                  help="Whether to display the screen during the game.")

flags.DEFINE_integer(name="episodes",
                     default=100,
                     help="Number of games that the agent plays.",
                     lower_bound=1)

flags.DEFINE_float(name="survived_step_reward",
                   default=0.1,
                   help="Whether and how much to reward the agent for each survived step.",
                   lower_bound=0.0)

flags.DEFINE_spaceseplist(name="mlp_hidden_units",
                          default=["256", "256"],
                          help="Number of units for each hidden layer.")

flags.DEFINE_float(name="learning_rate",
                   default=0.00005,
                   help="Size of the optimization step.",
                   lower_bound=0.0)

flags.DEFINE_float(name="discount_factor",
                   default=0.95,
                   help="How much future rewards are taken into account.",
                   lower_bound=0.0)

flags.DEFINE_float(name="epsilon",
                   default=0.8,
                   help="Epsilon value for epsilon-greedy.")

flags.DEFINE_float(name="eps_decrement",
                   default=0.99,
                   help="Factor of epsilon decrement per each step",
                   upper_bound=1.0)

flags.DEFINE_float(name="eps_minimum",
                   default=0.01,
                   help="Minimum value of epsilon.")

flags.DEFINE_integer(name="memory_size",
                     default=20_000,
                     help="Size of the experience replay buffer.")

flags.DEFINE_integer(name="min_memory",
                     default=3000,
                     help="Number of memories to gather before training.",
                     lower_bound=0)

flags.DEFINE_integer(name="seed",
                     default=42,
                     help="Random state for RNG.")

flags.DEFINE_integer(name="batch_size",
                     default=32,
                     help="Number of memories to train on per each step.",
                     lower_bound=1)

ACTION_MAP = {0: 0,
              1: 119}


def main(argv=None):

    if not FLAGS.display_screen:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    mlp_layers = tuple(int(dim) for dim in FLAGS.mlp_hidden_units)

    env = PLE(FlappyBird(),
              fps=30,
              display_screen=FLAGS.display_screen,
              frame_skip=1)
    
    agent = DQN(num_actions=2, # FlappyBird specific
                mlp_hidden_units=mlp_layers,
                learning_rate=FLAGS.learning_rate,
                discount_factor=FLAGS.discount_factor,
                epsilon=FLAGS.epsilon,
                eps_decrement=FLAGS.eps_decrement,
                eps_minimum=FLAGS.eps_minimum,
                memory_size=FLAGS.memory_size,
                min_memory=FLAGS.min_memory,
                seed=FLAGS.seed)

    total_steps = 0
    scores = []
    env.init()
    for ep in range(1, FLAGS.episodes + 1):
        if ep % 20 == 0:
            best_score = np.max(scores)
            best_ep = np.argmax(scores) + 1
            mean_score = np.mean(scores)
            logging.info(f"Current episode: {ep}")
            logging.info(f"Best score: {best_score:.2f}, from episode no. {best_ep}")
            logging.info(f"Mean score: {mean_score:.2f}")

        episode_score = 0
        done = False
        env.reset_game()
        state = list(env.getGameState().values())
        while not done:   
            total_steps += 1
            
            action = agent.predict(state)
            reward = env.act(ACTION_MAP[action])
            next_state = list(env.getGameState().values())
            done = env.game_over()
            
            if not done:
                reward += FLAGS.survived_step_reward

            episode_score += reward

            agent.extend_memory(state, action, reward, next_state, done)
            agent.fit(FLAGS.batch_size)        
        
            state = next_state
            
        scores.append(episode_score)
    logging.info("Done playing!")

if __name__ == "__main__":
    app.run(main)