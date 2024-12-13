# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=50000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=250,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.2,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.02,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=8,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """
    comments: Define Q-Network, inheriting from nn.Module.
    This network is composed of a linear layer, an activation function, 
    a linear layer, an activation function, and a linear layer.
    The input dimension is np.array(env.observation_space.shape).prod()(= 8).
    The output dimension is env.action_space.n(= 4).
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    comments: Linearly schedule the epsilon value for epsilon-greedy strategy.
    The epsilon value is decreased during training.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """comments: Set random seeds and define device for training"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """comments: Create an environment instance"""
    envs = make_env(args.env_id, args.seed)

    """
    comments: Initialize Q-network, optimizer, and target network.
    The parameters of target network is set to be the same as Q-network.
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """comments: Initialize the replay buffer"""
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """comments: This section is the main loop of the training process,
       which includes controlling the training duration, executing actions, 
       collecting experience, and updating the network"""
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        """comments: Adjust epsilon value according to the linear schedule with t = global_step."""
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """comments: Select action based on epsilon-greedy strategy."""
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """comments: do the action and get feedback"""
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """comments: Add the experience to the replay buffer."""
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """comments: Update the observation."""
        obs = next_obs if not dones else envs.reset()
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """comments: Sample a batch of experiences from the replay buffer"""
            data = rb.sample(args.batch_size)
            
            """comments: Compute the TD target values and the q_eval values, then get the loss of them."""
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """comments: Log training information."""
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """comments: Backpropagate to update network weights."""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """comments: update the target network when the number of global_step satisfies the condition."""
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    model_path_1 = "origin_model.pth"
    model_path_2 = "dqn_model.pth"

    # save the parameters of the q-network.
    torch.save(q_network.state_dict(), model_path_2)
    
    """close the env and tensorboard logger"""
    envs.close()
    writer.close()