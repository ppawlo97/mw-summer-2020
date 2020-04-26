"""Plays the game of FlappyBird with Deep Q Learning Agent."""
import json
import os
from pathlib import Path
from time import sleep

import numpy as np
from absl import app
from absl import flags
from absl import logging
from ple import PLE
from ple.games.flappybird import FlappyBird

from agent.DQN import DQN


FLAGS = flags.FLAGS

flags.DEFINE_bool(name="display_screen",
                  default=True,
                  help="Whether to display the screen during the game.")

flags.DEFINE_bool(name="train",
                  default=True,
                  help="Whether to train the agent or just play the game.")

flags.DEFINE_string(name="load_dir",
                    default=None,
                    help="Load already trained agent from given directory.\
                        Other params are then overwritten.")

flags.DEFINE_string(name="save_dir",
                    default=None,
                    help="Where to save trained agent.")

flags.DEFINE_string(name="summary_save_path",
                    default=None,
                    help="Where to save JSONL with episodes and agent summary.\
                        File should have jsonl extension.")

flags.DEFINE_integer(name="episodes",
                     default=500,
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
    
    if FLAGS.load_dir is not None:
        agent = DQN.load(model_dir=FLAGS.load_dir)
        if not FLAGS.train:
            agent._eps["epsilon"] = 0.0 # no more random actions
    else:
        logging.info("Creating new DQN agent.")
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
    
    scores = []
    pipes = []
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
        pipes_passed = 0
        done = False
        env.reset_game()
        state = list(env.getGameState().values())
        while not done:   
            
            action = agent.predict(state)
            reward = env.act(ACTION_MAP[action])
            next_state = list(env.getGameState().values())
            done = env.game_over()
            
            if not done:
                if reward == 1:
                    pipes_passed += 1
                reward += FLAGS.survived_step_reward

            episode_score += reward

            if FLAGS.train:
                agent.extend_memory(state, action, reward, next_state, done)
                agent.fit(FLAGS.batch_size)        
            else:
                sleep(0.01)

            state = next_state
            
        scores.append(episode_score)
        pipes.append(pipes_passed)
    
    logging.info("Done playing!")

    if FLAGS.summary_save_path is not None:
        summary_save_path = Path(FLAGS.summary_save_path)
        if summary_save_path.exists():
            logging.warning(f"Careful! Appending to already existing file {summary_save_path}")

        logging.info(f"Saving run summary to {summary_save_path}")
        with open(summary_save_path, "a") as f:
            summary = agent.get_config()
            summary["batch_size"] = FLAGS.batch_size
            summary["episodes"] = FLAGS.episodes
            summary["survived_step_reward"] = FLAGS.survived_step_reward
            summary["scores"] = scores
            summary["pipes"] = pipes
            json.dump(summary, f)
            f.write("\n")

    if FLAGS.save_dir is not None:
        agent.save(FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)