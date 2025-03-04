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

# Define the training steps and log frequency
training_steps = 1 * 10**7
log_freq = training_steps // 100 


class SimpleTrainingLogger(BaseCallback):
    def __init__(self, log_freq=1000, log_file="training_log.csv"):
        super().__init__()
        self.log_freq = log_freq
        self.log_file = log_file
        self.interval_data = []

        # Current episode tracking
        self.current_reward = 0
        self.current_length = 0

        # Aggregate metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        self.start_time = time.time()
        self.interval_data = []

    def _on_step(self):
        # Get environment information
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]

        # Accumulate current episode data
        self.current_reward += reward
        self.current_length += 1

        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.current_reward = 0
            self.current_length = 0

        # Log at specified intervals
        if self.num_timesteps % self.log_freq == 0:
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            best_reward = np.max(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

            self.interval_data.append({
                "Total Steps": self.num_timesteps,
                "Total Time": time.time() - self.start_time,
                "Avg Reward": avg_reward,
                "Best Reward": best_reward,
                "Avg Length": avg_length
            })

            # Save to CSV
            pd.DataFrame(self.interval_data).to_csv(self.log_file, index=False)

            # Reset metrics for next interval
            self.episode_rewards = []
            self.episode_lengths = []

        return True

def train_model(params, resume=False):
    net_arch = params.get('net_arch', [64, 64])
    training_steps = params.get('training_steps', 1 * 10**4)
    gamma = params.get('gamma', 0.99)
    ent_coef = params.get('ent_coef', 0.01)
    log_file = params.get('log_file', "training_log.csv")
    checkpoint_path = params.get('checkpoint_path', "./ppo_flappybird/")

    # Initialize environment and model
    env = DummyVecEnv([lambda: gym.make("FlappyBird-v0", use_lidar=False)])
    if resume and os.path.exists(f"{checkpoint_path}/best_model.zip"):
        model = PPO.load(f"{checkpoint_path}/best_model", env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, gamma=gamma, ent_coef=ent_coef, policy_kwargs={'net_arch': net_arch})

    # Train with callbacks
    logger = SimpleTrainingLogger(log_freq=log_freq, log_file=log_file)
    checkpoint = CheckpointCallback(save_freq=log_freq, save_path=checkpoint_path)
    model.learn(total_timesteps=training_steps, callback=[logger, checkpoint])
    model.save(f"{checkpoint_path}/final_model")

def plot_training_progress(log_files, labels):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))  # Generate distinct colors
    for log_file, label, color in zip(log_files, labels, colors):
        df = pd.read_csv(log_file)
        best_rewards = df["Best Reward"].cummax()  # Cumulative maximum of best rewards
        plt.plot(df["Total Steps"], best_rewards, label=f"{label} Best Score", color=color, alpha=0.6)
        plt.plot(df["Total Steps"], df["Avg Reward"], label=f"{label} Average Score", color=color, linestyle='--', alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.legend()
    plt.show()

# Example usage

#### baseline
param1={'net_arch': [64, 64], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.00, 'log_file': "param1.csv", 'checkpoint_path': "./ppo_flappybird_param1/"}

#### change gamma
param2={'net_arch': [64,64], 'training_steps': training_steps, 'gamma': 0.95, 'ent_coef': 0.00, 'log_file': "param2.csv", 'checkpoint_path': "./ppo_flappybird_param2/"}
param3={'net_arch': [64,64], 'training_steps': training_steps, 'gamma': 0.90, 'ent_coef': 0.00, 'log_file': "param3.csv", 'checkpoint_path': "./ppo_flappybird_param3/"}
param4={'net_arch': [64,64], 'training_steps': training_steps, 'gamma': 0.85, 'ent_coef': 0.00, 'log_file': "param4.csv", 'checkpoint_path': "./ppo_flappybird_param4/"}


#### change architecture
param5={'net_arch': [128, 128], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.00, 'log_file': "param5.csv", 'checkpoint_path': "./ppo_flappybird_param5/"}
param6={'net_arch': [64, 64, 64], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.00, 'log_file': "param6.csv", 'checkpoint_path': "./ppo_flappybird_param6/"}
param7={'net_arch': [32, 64, 32], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.00, 'log_file': "param7.csv", 'checkpoint_path': "./ppo_flappybird_param7/"}
param11={'net_arch': [16, 16], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.00, 'log_file': "param11.csv", 'checkpoint_path': "./ppo_flappybird_param11/"}

#### change ent_coef
param8={'net_arch': [64, 64], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.01, 'log_file': "param8.csv", 'checkpoint_path': "./ppo_flappybird_param8/"}
param9={'net_arch': [64, 64], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.02, 'log_file': "param9.csv", 'checkpoint_path': "./ppo_flappybird_param9/"}
param10={'net_arch': [64, 64], 'training_steps': training_steps, 'gamma': 0.99, 'ent_coef': 0.03, 'log_file': "param10.csv", 'checkpoint_path': "./ppo_flappybird_param10/"}

### final list of params$
param12={'net_arch': [128, 128,128], 'training_steps': training_steps, 'gamma': 0.9, 'ent_coef': 0.01, 'log_file': "param12.csv", 'checkpoint_path': "./ppo_flappybird_param12/"}


params_list = [
    param12
]



for i in range(len(params_list)):
    train_model(params_list[i], resume=False)
# log_files = ["param1.csv", "param2.csv"]
# labels = ["Params Set 1", "Params Set 2"]
# plot_training_progress(log_files, labels)
