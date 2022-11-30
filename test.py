import gym
from gym.wrappers import RecordVideo

# gym("Humanoid-v4", render_mode="rgb_array")
env = gym.make("Humanoid-v4", render_mode="rgb_array")

env = RecordVideo(env, "video/")

for _ in range(10):
    rewards = 0
    steps = 0
    done = False
    _, _ = env.reset()

    while not done:
        env.start_video_recorder()
        env.unwrapped.render()
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        steps += 1
        rewards += reward

    print("Testing steps: {} rewards {}: ".format(steps, rewards))

env.close()