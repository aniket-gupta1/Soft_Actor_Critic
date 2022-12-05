import argparse
import os
import glob
from moviepy.editor import *
from natsort import natsorted
from agent import *
from model import *

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

class Play():
    def __init__(self, cfg, env, weights_path):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(cfg).to(self.device)
        self.weights_path = weights_path
        self.video_path = cfg.video_path

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy(state)

        return action.cpu().numpy().reshape(-1)

    def make_video(self, path, n=1):
        env = RecordVideo(self.env, path)

        for i in range(n):
            rewards = 0
            steps = 0
            done = False
            observation, _ = env.reset()
            env.start_video_recorder()

            while True:
                action = self.act(observation)
                observation, reward, done, _, _ = env.step(action)
                steps += 1
                rewards += reward

                if steps >= 1000 or done:
                    break

            print("-" * 80)
            print("Episode steps: {} || Rewards {}: ".format(steps, rewards))
            print("-" * 80)

        env.close()

    def run(self):
        weights_path = os.listdir(self.weights_path)
        policy_weights_path = [path for path in weights_path if path[-10:]=="policy.pth"]

        for i, policy_path in enumerate(policy_weights_path):
            self.policy.load(os.path.join(self.weights_path, policy_path))
            self.make_video(self.video_path + f"/epoch_{i}/", 1)

    def combine_video(self):
        video_paths = sorted(os.listdir(self.video_path))
        L = []

        for path in video_paths:
            clip_path = os.path.join(self.video_path, path, "rl-video-episode-0.mp4")
            clip = VideoFileClip(clip_path)
            L.append(clip)

        video = concatenate_videoclips(L)
        video.to_videofile(os.path.join(self.video_path, "OUTPUT.mp4"), fps=10, remove_temp=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Humanoid-v4")
    parser.add_argument('--path', type=str, default="/home/ngc/NEU_Courses/data/data/Humanoid-v4_model_weights")
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

    play = Play(cfg, env, args.path)
    play.run()
    play.combine_video()




