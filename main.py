import torch
import torch.nn as nn
import os
import argparse
import time
import gym
import yaml
from agent import *
from easydict import EasyDict

class config(object):
    def __init__(self, env):
        super(config, self).__init__()
        self.data_dir = "./data"
        if not os.path.exists(self.data_dir):
            os.system(f"mkdir {self.data_dir}")

        self.epochs = 10000
        self.batch_size = 256
        self.lr = 0.0003
        self.save_dir = f"./data/{str(env)}_model_weights/"
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
        self.eval_interval = 100

        self.log_dir = f"./data/{str(env)}_runs/"

        self.video_episodes = 10
        self.video_path = f"./data/{str(env)}_video"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Humanoid-v4")
    parser.add_argument('--use_ckp', type=bool, default=False)
    parser.add_argument('--vid_eps', type=int, default=10)
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

    cfg.video_episodes = args.vid_eps

    agent = SACAgent(env, cfg)
    agent.run()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', type=str, default="./conf/Humanoid.yaml")
    # args = parser.parse_args()
    #
    # with open(args.conf) as file:
    #     cfg = EasyDict(yaml.safe_load(file))
    #
    # env = gym.make(cfg.env.name, render_mode="rgb_array")
    #
    # # Change config params according to env states and actions
    # cfg.env.state_shape = env.observation_space.shape
    # cfg.env.action_shape = env.action_space.shape
    # cfg.policy.input_dim = env.observation_space.shape[0]
    # cfg.policy.output_dim = env.action_space.shape[0] * 2
    # cfg.policy.hidden_units = [256, 256]
    #
    # cfg.critic.input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    # cfg.critic.output_dim = 1
    # cfg.critic.hidden_units = [256, 256]




