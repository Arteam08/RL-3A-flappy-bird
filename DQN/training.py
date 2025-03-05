import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import randint
import time
import os
import model
#Environment
env = gymnasium.make("FlappyBird-v0")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Parameter
BUFFER_SIZE = 757624   # replay buffer size
BATCH_SIZE = 228             # minibatch size
GAMMA = 0.8777525034865641                # discount factor
TAU = 0.005566248249524419                      # for soft update of target parameters
UPDATE_EVERY = 3         # how often to update the network



# Init agent
agent = model.Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)
checkpoint_path = "checkpoints/model_checkpoint"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)  

def DQN(n_episodes=100000, eps_start=0.9, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning Training
    """
    scores = []  
    scores_window = deque(maxlen=100)  #100 scores
    eps = eps_start
    best_score = 1


    for i_episode in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

        scores.append(score)
        scores_window.append(score)
        eps = max(eps * eps_decay, eps_end)  #Decrease eps

        # Save highest checkpoint
        score_avg = np.mean(scores_window)
        if score_avg > best_score:
            best_score = score_avg
            torch.save(agent.qnetwork_local.state_dict(), f"{checkpoint_path}_best_{score_avg:.2f}.pth")
            print(f" New best model saved! Score: {score_avg:.2f}")

        # Save checkpoint every 1000 eps
        if i_episode % 1000 == 0:
            torch.save(agent.qnetwork_local.state_dict(), f"{checkpoint_path}_{i_episode}_{score_avg:.2f}.pth")
            print(f"Checkpoint saved at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f'\rEpisode {i_episode}\tAverage Score: {score_avg:.2f}', end="")

        # Early stop
        if score_avg >= 100:
            print(f'\n✅ Environment solved in {i_episode} episodes! Avg Score: {score_avg:.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'final_model.pth')
            break

    env.close()  
    return scores

scores = DQN()