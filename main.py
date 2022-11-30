import torch
import torch.nn as nn
import os
import argparse
import time
import gym
from agent import *

class config(object):
    def __init__(self, env):
        super(config, self).__init__()

        self.epochs = 10000
        self.batch_size = 256
        self.lr = 0.0003
        self.save_dir = f"./{str(env)}_model_weights/"
        if not os.path.exists(self.save_dir):
            os.system(f"mkdir {self.save_dir}")

        self.exploration_epochs = 2000
        self.grad_clip = None

        # memory
        self.capacity = 1000000

        # Critic Network
        self.input_dim = 12
        self.hidden_units = [1,2]
        self.output_dim = 12

        # Policy network
        self.policy_input_dim = 12
        self.policy_hidden_units = [1,2]
        self.policy_output_dim = 12

        # Environment info
        self.state_shape = None
        self.action_shape = None


        self.network = 'fcn'
        self.memory_size = 1e6
        self.gamma = 0.99
        self.tau = 0.005
        self.entropy_tuning = True
        self.per = False
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_annealing = 0.0001
        self.eval_interval = 10000

        self.log_dir = f"./{str(env)}_runs/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Humanoid-v4")
    parser.add_argument('--use_ckp', type=bool, default=False)
    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode="rgb_array")

    cfg = config(args.env_name)
    cfg.policy_input_dim = env.observation_space.shape[0]
    cfg.policy_output_dim = env.action_space.shape[0] * 2
    cfg.policy_hidden_units = [256, 256]

    cfg.input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    cfg.output_dim = 1
    cfg.hidden_units = [256, 256]

    cfg.state_shape = env.observation_space.shape
    cfg.action_shape = env.action_space.shape

    agent = SACAgent(env, cfg)
    agent.run()




