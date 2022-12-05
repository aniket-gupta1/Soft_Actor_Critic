import yaml
import argparse
from easydict import EasyDict
import os
import glob
from moviepy.editor import *

def combine_video(video_path):
    video_paths = sorted(os.listdir(video_path))
    L = []

    for path in video_paths:
        clip_path = os.path.join(video_path, path, "rl-video-episode-0.mp4")
        clip = VideoFileClip(clip_path)
        L.append(clip)

    video = concatenate_videoclips(L)
    video.to_videofile(os.path.join(video_path, "OUTPUT.mp4"), fps=10, remove_temp=False)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', type=str, default="./conf/Humanoid.yaml")
    # args = parser.parse_args()
    #
    # with open(args.conf) as file:
    #     cfg = EasyDict(yaml.safe_load(file))
    #
    # print(cfg.env.name)
    # cfg.env.name = "aniket"
    # print(cfg.env.name)

    combine_video("/home/ngc/NEU_Courses/Project/data/Humanoid-v4_video")

