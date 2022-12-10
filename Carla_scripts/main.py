import torch
import torch.nn as nn
import os
import argparse
import time
import gym
import yaml
from agent import *
from easydict import EasyDict
from carla_env import carla_env


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

        self.exploration_epochs = 5
        self.grad_clip = None

        # memory
        self.capacity = 1000

        # Critic Network
        self.action_dim = 3
        self.hidden_units = None
        self.output_dim = 1

        # Policy network
        self.policy_input_dim = None
        self.policy_hidden_units = None
        self.policy_output_dim = None

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

        self.motion_batch = 8


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Carla")
    parser.add_argument('--use_ckp', type=bool, default=True)
    parser.add_argument('--vid_eps', type=int, default=10)
    args = parser.parse_args()

    cfg = config(args.env_name)
    env = carla_env(motion_batch=cfg.motion_batch)

    cfg.policy_input_dim = env.observation_space.shape[0]
    cfg.policy_output_dim = env.action_space.shape[0] * 2
    cfg.policy_hidden_units = [256, 256]

    cfg.input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    cfg.output_dim = 1
    cfg.hidden_units = [256, 256]

    cfg.state_shape = (cfg.motion_batch, env.observation_space.shape[0], env.observation_space.shape[1])
    cfg.action_shape = env.action_space.shape

    cfg.video_episodes = args.vid_eps

    agent = SACAgent(env, cfg)
    if args.use_ckp:
        agent.policy.load(os.path.join(cfg.save_dir), "latest_policy.pth")
        agent.critic.load(os.path.join(cfg.save_dir), "latest_critic.pth")
        agent.critic_target.load(os.path.join(cfg.save_dir), "latest_critic_target.pth")

    agent.run()




