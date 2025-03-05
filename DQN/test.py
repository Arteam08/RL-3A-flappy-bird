import numpy as np
import gymnasium as gym
import torch
import model
import pygame
pygame.quit()  
pygame.init()  

#Environment
env = gym.make("FlappyBird-v0", use_lidar = False)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

agent = model.Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

saved_model="model_checkpoint_best_26.88.pth" # Load your best model here
agent.qnetwork_target.load_state_dict(torch.load(saved_model))
# Load Model
agent.qnetwork_local.load_state_dict(torch.load(saved_model))
print("✅ Done!")

num_time_steps=15 #Play 15 times
for i in range(num_time_steps):
    state = env.reset()[0]  
    state = np.array(state, dtype=np.float32).reshape(1, -1)
    state = np.array(state, dtype=np.float32)  
    done = False
    total_reward = 0

    while not done:
        env.render()  
        

        
        action = agent.act(state, eps=0)  
        
        
        next_state, reward, done, _,pipe = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, -1)
        next_state = np.array(next_state, dtype=np.float32)  
    
        state = next_state
        total_reward += reward


    print(f"Episode {i+1}: Total Reward = {total_reward}")
    print("Pipes passed"+ str(pipe["score"]))
    

env.close()  
