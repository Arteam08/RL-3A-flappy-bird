import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_progress(log_files, labels):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))  # Generate distinct colors
    for log_file, label, color in zip(log_files, labels, colors):
        df = pd.read_csv(log_file)
        best_rewards = df["Best Reward"].cummax()  # Cumulative maximum of best rewards
        plt.plot(df["Total Steps"], best_rewards, label=f"{label} Best Score", color=color, alpha=0.6)
        # plt.plot(df["Total Steps"], df["Avg Reward"], label=f"{label} Average Score", color=color, linestyle='--', alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.legend()
    # plt.savefig("entropy comparison.png")
    plt.show()

# def plot_training_progress_time(log_files, labels):
#     plt.figure(figsize=(10, 5))
#     colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))  # Generate distinct colors
#     for log_file, label, color in zip(log_files, labels, colors):
#         df = pd.read_csv(log_file)
#         best_rewards = df["Best Reward"].cummax()  # Cumulative maximum of best rewards
#         plt.plot(df["Total Time"], best_rewards, label=f"{label} Best Score", color=color, alpha=0.6)
#         # plt.plot(df["Total Time"], df["Avg Reward"], label=f"{label} Average Score", color=color, linestyle='--', alpha=0.6)
    
#     plt.xlabel("Computation Time (seconds)")
#     plt.ylabel("Score")
#     plt.title("Training Progress Over Time")
#     plt.legend()
#     plt.show()

log_files_gamma = ["param1.csv", "param2.csv", "param3.csv", "param4.csv"]
labels_gamma = [r"$\gamma=0.99$", r"$\gamma=0.95$", r"$\gamma=0.90$", r"$\gamma =0.85$"]

log_files_layers=["param11.csv","param1.csv", "param5.csv", "param6.csv", "param7.csv"]
labels_layers=[r"2 layers 16 units",r"2 layers 64 units", r"2 layers 128 units", r"3 layers 64 units", r"3 layers 32 units" ]

log_files_ent=["param1.csv", "param8.csv", "param9.csv", "param10.csv"]
labels_ent=[r"Entropy Coef = 0.0", r"Entropy Coef = 0.01", r"Entropy Coef = 0.02", r"Entropy Coef = 0.03"]

# plot_training_progress(log_files, labels)
plot_training_progress(["param12.csv"], ["final"])