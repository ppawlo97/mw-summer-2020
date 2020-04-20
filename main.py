import os

import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird

from DQN import DQN

# os.environ["SDL_VIDEODRIVER"] = "dummy"


ACTION_MAP = {0: 0,
              1: 119}

env = PLE(FlappyBird(),
          fps=30,
          display_screen=True,
          frame_skip=2)
agent = DQN(2)


scores = []
env.init()
for i in range(5000):
    if i % 20 == 0:
        print(i, np.nanmean(scores[-100:]))
    
    env.reset_game()
    episode_score = 0
    done = False
    state = list(env.getGameState().values())
    while not done:   
        action = agent.predict(state)
        reward = env.act(ACTION_MAP[action])
        next_state = list(env.getGameState().values())
        done = env.game_over()
        
        if done:
            reward = -1000
        episode_score += reward

        agent.extend_memory(state, action, reward, next_state, done)
        state = next_state
        agent.fit(64)        
    
    scores.append(episode_score)
