import os
import random
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import gym
import torch
import numpy as np
from dqn import QNetwork, make_env

# load the model
env = make_env("LunarLander-v2", seed=30)
q_network = QNetwork(env)
model_path = "dqn_model.pth"
q_network.load_state_dict(torch.load(model_path))
q_network.eval()

# test the model
num_episodes = 50
reward_list = []
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            q_values = q_network(torch.Tensor(obs).unsqueeze(0))
            action = torch.argmax(q_values, dim=1).item()
        
        next_obs, reward, done, info = env.step(action)

        # env.render()
        
        total_reward += reward
        
        obs = next_obs
    
    reward_list.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

print(f"Mean Reward = {np.mean(reward_list)}")