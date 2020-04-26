"""Runs FlappyBird multiple times with only one parameter changed."""
import subprocess
from time import time

from absl import app
from absl import logging

LOGS_PATH = "./params_search.jsonl"

PARAMS = {"survived_step_reward": [0.0, 2, 10],
          "mlp_hidden_units": ["128 128", "128 64 32", "256 128 64"],
          "learning_rate": [0.005, 0.0005, 0.000005],
          "discount_factor": [0.5, 0.8, 0.99],
          "epsilon": [0.3, 0.5, 1],
          "eps_minimum": [0.001, 0.1],
          "memory_size": [5000, 10_000, 50_000],
          "min_memory": [0, 5000, 10_000],
          "batch_size": [16, 64, 128]}


def main(argv=None):
    # run on 500 episodes by default
    i = 1
    script_start = time()
    for key in PARAMS:
        for param in PARAMS[key]:
            cmd = f"python run_flappybird.py --nodisplay_screen --summary_save_path='{LOGS_PATH}' --{key}={param}"
            logging.info(f"Starting run no. {i}...")
            start = time()
            subprocess.run(cmd, shell=True)
            
            time_passed = (time() - start) / 60
            logging.info(f"Finished in {time_passed:.2f} minutes...")
            i += 1
        
        logging.info(f"Done checking {key}...")

    total_time = (time() - script_start) / 60 
    logging.info(f"Finished everything in {total_time:.2f} minutes...")


if __name__ == "__main__":
    app.run(main)