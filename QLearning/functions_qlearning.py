import optuna
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import randint
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import time
import os
import json
import random

import gymnasium as gym
import math
import numpy as np
import pandas as pd
from pathlib import Path

from typing import cast, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from tqdm.notebook import tqdm

import pickle

def save_variable_in_pickle(data, name_file):
    with open(name_file, "wb") as f:
        pickle.dump(data, f)

def load_variable_pickel(name_file):
    # Charger les données
    with open(name_file, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

class CustomRewardWrapperFlappyBird(gym.Wrapper):
    """custom reward wrapper to change the reward of the gymnasium environement of Flappy Bird
        Before
        +0.1 - every frame it stays alive
        +1.0 - successfully passing a pipe
        -1.0 - dying
        −0.5 - touch the top of the screen
        After
        if reward_extreme=True
        Reward dead : -1000
        Reward alive : 0
        if reward_moderate=True
        Penalty more important for death and reward more important for passing pipes.
    """
    def __init__(self, env, reward_extreme=False, reward_moderate=False):
        super().__init__(env)
        self.reward_extreme = reward_extreme
        self.reward_moderate = reward_moderate


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # version extreme
        if self.reward_extreme:
            if done:
                reward = -1000
            else: 
                reward = 0

        if self.reward_moderate:
            if reward== 1 :
                reward = 2  # On augmente la récompense des tuyaux
            elif reward == 0.1:
                reward = 0  # Chaque frame survivante reste inchangée
            elif reward == -1.0:
                reward = -5  # Pénalité pour mourir (2 fois et demi plus grave que passer un pipe)
            elif reward == -0.5:
                reward = 0  # Pénalité pour toucher le haut de l'écran
        
        return obs, reward, done, truncated, info
    

def extract_state(obs, player_x=0):
    """
    Extracts the variables needed for Q-learning from the observation.
    
    Variables returned:
    - x0: Horizontal distance between player and next lower pipe.
    - y0: Vertical distance from player to bottom of next pipe.
    - vel: Player's vertical speed.
    - y1: Vertical distance between player and next lower pipe.
    
    :param obs: np.array containing environmental observations.
    :param player_x: Player's horizontal position (often 0 if fixed).
    :return: (x0, y0, vel, y1)
    """
    # horizontal distance to next pipe (x0)
    x0 = obs[3] - player_x # next pipe horizontal pos - player horizontal pos

    # Vertical distance to next pipe bottom (y0)
    y0 = obs[9] - obs[5] # player vertical pos - next bottom pipe vertical pos

    # player vertical speed (vel)
    vel = obs[10]  

    # vertical distance to bottom of next pipe (y1)
    y1 = obs[9] - obs[8] # player vertical pos - next next bottom pipe vertical pos

    return x0, y0, vel, y1

class CustomObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to modify the observation space of Flappy Bird.
    """
    def __init__(self, env, discretize=False, bins=20, simplify_obs=False):
        super().__init__(env)
        self.discretize = discretize  # Discretization flag
        self.bins = bins  # Number of bins for discretization
        self.simplify_obs = simplify_obs

    def observation(self, obs):
        """
        Extracts and discretizes observation space
        """
        # Extraction de 3 valeurs : distance horizontale au prochain pipe, 
        # la distance verticale au haut de next pipe, la distance verticale au bas de next pipe

        new_obs=obs
        
        if self.simplify_obs:
            new_obs = extract_state(obs) # x0, y0, vel, y1

        # Discrétisation pour Q-learning
        if self.discretize:
            new_obs = np.digitize(new_obs, np.linspace(-1, 1, self.bins)) - 1  # Convertir en indices discrets

        return new_obs

# Initialiser l'environnement
def define_discrete_flappy_bird(change_reward=True, bins=25, simplify_obs=False, reward_extreme=True, reward_moderate=False):
    env = gym.make("FlappyBird-v0", use_lidar=False)
    if change_reward:
        env = CustomRewardWrapperFlappyBird(env, reward_extreme=reward_extreme, reward_moderate=reward_moderate) # Reward changée
    env = CustomObservationWrapper(env, discretize=True, bins=bins, simplify_obs=simplify_obs)  # Discrétisation activée
    return env

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

DISPLAY_EVERY_N_EPISODES = 50

def get_q_values(state, q_table, num_actions):
    """Retourne les Q-values associées à un état sous forme de tuple."""
    state = tuple(state)  # Convertit l'état en tuple pour être utilisé comme clé
    if state not in q_table:
        q_table[state] = [0.0] * num_actions  # Initialisation des Q-values
    return q_table[state]


def epsilon_greedy_policy_q_dict(state, q_table, epsilon, num_actions=2):
    """Choisit une action selon la politique epsilon-greedy."""
    state = tuple(state)  # Convertit en tuple
    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)  # Exploration entre 0 et 1
    else:
        action = np.argmax(get_q_values(state, q_table, num_actions))  # Exploitation
    return action

def greedy_policy_q_dict(state, q_table, num_actions=2):
    """Choisit l'action avec la meilleure Q-value (greedy policy)."""
    state = tuple(state)  # Convertit en tuple
    return np.argmax(get_q_values(state, q_table, num_actions))

def q_learning(
    environment: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.5,
    num_episodes: int = 60000,
) -> Dict:
    """
    Q-learning avec une Q-table stockée dans un dictionnaire (sans sauvegarde de fichier).
    """
    q_table = {}  # Dictionnaire pour stocker les Q-values
    num_actions = environment.action_space.n  # Nombre d'actions possibles
    history_score=[]
    history_reward=[]

    for episode_index in tqdm(range(1, num_episodes + 1)):
        obs, _ = environment.reset()
        state = tuple(obs)  # Convertir l'observation en tuple (clé du dictionnaire)
        terminated = False
        cumulated_reward = 0

        while not terminated:
            # Politique epsilon-greedy
            action = epsilon_greedy_policy_q_dict(state, q_table, epsilon, num_actions)

            # Prendre une action
            obs_prime, reward, terminated, _, info = environment.step(action)
            next_state = tuple(obs_prime)  # Convertir en clé

            # Récupérer les Q-values
            q_values = get_q_values(state, q_table, num_actions)
            next_q_values = get_q_values(next_state, q_table, num_actions)  # Assure que l'état existe

            # Mise à jour de la Q-table (règle de Q-learning)
            q_values[action] += alpha * (reward + gamma * max(next_q_values) - q_values[action])

            # Mise à jour de l'état actuel
            state = next_state

            cumulated_reward += reward
        
        if episode_index % (num_episodes // 10) == 0:
            last_10_percent_scores = history_score[-(num_episodes // 10):]  # Prend les derniers 10% des scores
            print(f"Épisode {episode_index}/{num_episodes} - Moyenne cumulée {np.mean(history_score)} - Moyenne des derniers 10% : {np.mean(last_10_percent_scores)}")

        history_score.append(info['score'])
        history_reward.append(cumulated_reward)
        
        # Réduction progressive de epsilon
        epsilon = max(0.01, epsilon * 0.995)  # Diminue progressivement epsilon mais reste ≥ 0.01

    return history_score, history_reward, q_table

def plot_history_score(history_score, window=50, title="Flappy Bird - Simplified game - Evolution of scores with training"):
    """
    Affiche l'évolution des scores au fil des épisodes en RL.
    
    - Trace les scores de chaque épisode.
    - Affiche une moyenne pondérée des scores (moyenne glissante).
    - Indique le score maximal atteint à chaque épisode.
    
    :param history_score: Liste des scores obtenus à chaque épisode.
    :param window: Taille de la fenêtre pour la moyenne glissante.
    """
    num_episodes = len(history_score)
    episodes = np.arange(1, num_episodes + 1)

    avg_scores = np.convolve(history_score, np.ones(window)/window, mode='valid')

    max_scores = np.maximum.accumulate(history_score)

    plt.figure(figsize=(12, 6))
    
    sns.lineplot(x=episodes, y=history_score, label="Score by episode", alpha=0.3, color='blue')

    sns.lineplot(x=episodes[window-1:], y=avg_scores, label=f"Mean ({window} episodes)", color='orange')

    # Tracer le maximum des scores atteints
    sns.lineplot(x=episodes, y=max_scores, label="Maximum score", color='red')

    # Labels et titre
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_history_reward(history_reward, window=50, title="Flappy Bird - Simplified game - Evolution of reward with training"):
    """
    Affiche l'évolution des reward au fil des épisodes en RL.

    :param history_reward: Liste des reward obtenus à chaque épisode.
    :param window: Taille de la fenêtre pour la moyenne glissante.
    """
    num_episodes = len(history_reward)
    episodes = np.arange(1, num_episodes + 1)

    avg_reward = np.convolve(history_reward, np.ones(window)/window, mode='valid')

    max_reward = np.maximum.accumulate(history_reward)

    plt.figure(figsize=(12, 6))
    
    sns.lineplot(x=episodes, y=history_reward, label="Reward by episode", alpha=0.3, color='blue')

    #sns.lineplot(x=episodes[window-1:], y=avg_reward, label=f"Mean ({window} episodes)", color='orange')

    sns.lineplot(x=episodes, y=max_reward, label="Maximum reward", color='red')

    # Labels et titre
    plt.xlabel("Episodes")
    plt.ylabel("reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

