# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import random
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, 
                 all_actions,
                 num_states,
                 alpha = 0.1,
                 gamma = 0.9,
                 epsilon=1.0,
                 min_epsilon = 0.01,
                 epsilon_decay = 0.99):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_states,len(all_actions)))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.all_actions)
        else:
            q_values = self.q_table[observation]
            return int(np.argmax(q_values))
    
    def learn(self, state, action, reward, next_state, next_action):
        """learn from experience"""
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        # Decay epsilon
        # if self.epsilon > self.min_epsilon:
        #     self.epsilon *= self.epsilon_decay
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, 
                 all_actions,
                 num_states,
                 alpha = 0.1,
                 gamma = 0.9,
                 epsilon=1.0,
                 min_epsilon = 0.01,
                 epsilon_decay = 0.99):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_states,len(all_actions)))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.all_actions)
        else:
            q_values = self.q_table[observation]
            return int(np.argmax(q_values))
    
    def learn(self, state, action, reward, next_state):
        """learn from experience"""
        best_next_action = np.argmax(self.q_table[next_state])
        sample = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] = self.alpha * sample + (1 - self.alpha) * self.q_table[state][action]

        # if self.epsilon > self.min_epsilon:
        #     self.epsilon *= self.epsilon_decay
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####
    
    

class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(self, 
                 all_actions,
                 num_states,
                 alpha = 0.1,
                 gamma = 0.9,
                 epsilon=1.0,
                 min_epsilon = 0.01,
                 epsilon_decay = 0.99,
                 num_of_steps = 10):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_states,len(all_actions)))
        self.model = {} # (state, action) -> (next_state, next_action)
        self.num_of_steps = num_of_steps

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.all_actions)
        else:
            q_values = self.q_table[observation]
            return int(np.argmax(q_values))
    
    def learn(self, state, action, reward, next_state):
        """learn from experience"""
        best_next_action = np.argmax(self.q_table[next_state])
        sample = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] = self.alpha * sample + (1 - self.alpha) * self.q_table[state][action]

        self.model[(state, action)] = (next_state, reward)

        for _ in range(self.num_of_steps):
            (s, a), (s_, r) = random.choice(list(self.model.items()))
            best_next_action = np.argmax(self.q_table[s_])
            sample = r + self.gamma * self.q_table[s_][best_next_action]
            self.q_table[s][a] = self.alpha * sample + (1 - self.alpha) * self.q_table[s][a]

        # if self.epsilon > self.min_epsilon:
        #     self.epsilon *= self.epsilon_decay
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####
