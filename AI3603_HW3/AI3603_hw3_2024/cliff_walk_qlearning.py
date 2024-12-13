# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import matplotlib.pyplot as plt
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions) # [0,1,2,3]
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####

# construct the intelligent agent.
agent = QLearningAgent(all_actions, num_states=48)

episode_rewards = []
epsilons = []

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # env.render()
        # update the episode reward
        episode_reward += r
        # print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, a, r, s_)
        s = s_
        if isdone:
            time.sleep(0.1)
            break

    if agent.epsilon > agent.min_epsilon:
        agent.epsilon *= agent.epsilon_decay

    episode_rewards.append(episode_reward)
    epsilons.append(agent.epsilon)
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
print('\ntraining over\n')   

# close the render window after training.
env.close()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epsilons, label='Epsilon', color='orange')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Over Time')
plt.legend()

plt.tight_layout()
plt.show()

def test_agent(agent, env, episodes=1, render=True):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(0.2)  # Slow down the rendering
            action = agent.choose_action(state)  # Choose action without exploration (epsilon=0)
            state, reward, done, info = env.step(action)
        if render:
            env.render()
            time.sleep(0.2)  # Pause at the end of the episode
    env.close()

# Set epsilon to 0 to ensure the agent only exploits learned knowledge
agent.epsilon = 0.0
test_agent(agent, env, episodes=1, render=True)

##### END CODING HERE #####