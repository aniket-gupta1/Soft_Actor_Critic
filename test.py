import yaml
import argparse
from easydict import EasyDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./conf/Humanoid.yaml")
    args = parser.parse_args()

    with open(args.conf) as file:
        cfg = EasyDict(yaml.safe_load(file))

    print(cfg.env.name)
    cfg.env.name = "aniket"
    print(cfg.env.name)

