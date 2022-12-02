import os
import time
import gym
from model import *
from memory import *
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import RecordVideo

class SACAgent(object):
    def __init__(self, env, cfg):
        super(SACAgent, self).__init__()
        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(cfg).to(self.device)
        self.critic = ACNetwork(cfg).to(self.device)
        self.critic_target = ACNetwork(cfg).to(self.device)

        # Copy parameters of the learning network to the target network
        self.hard_update(self.critic_target, self.critic)
        # Disable gradient calculations of the target network
        self.grad_false(self.critic_target)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.critic_q1_optim = torch.optim.Adam(self.critic.Q1.parameters(), lr=cfg.lr)
        self.critic_q2_optim = torch.optim.Adam(self.critic.Q2.parameters(), lr=cfg.lr)

        self.memory = ReplayMemory(cfg, self.device)

        self.step_count = 0
        self.writer = SummaryWriter()

        self.epochs = cfg.epochs
        self.save_dir = cfg.save_dir
        self.exploration_epochs = cfg.exploration_epochs

        self.gamma = cfg.gamma
        self.alpha = cfg.alpha
        self.tau = cfg.tau
        self.grad_clip = cfg.grad_clip

        self.eval_interval = cfg.eval_interval
        self.eval_step = 0
        self.render = False

        # Entropy tuning params
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        # Video
        self.video_path = cfg.video_path
        self.video_episodes = cfg.video_episodes

    def run(self):
        for epoch in range(self.epochs):
            self.train_episode(epoch)
            if epoch % self.eval_interval == 0:
                self.evaluate()
                self.save_models(epoch)
                self.make_video(self.video_path + f"/epoch_{epoch}/", self.video_episodes)

    def train_episode(self, epoch):
        episode_reward = 0
        episode_steps = 0
        state, info = self.env.reset()

        while True:
            action = self.act(epoch, state)
            next_state, reward, done, _, _ = self.env.step(action)

            episode_reward += reward
            episode_steps += 1

            self.memory.append(state, action, reward, next_state, done)

            # Check if memory can provide a batch and that we have done exploration epochs
            if self.memory.can_provide_sample() and epoch >= self.exploration_epochs:
                self.soft_update(self.critic_target, self.critic, self.tau)

                batch = self.memory.sample()
                q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.critic_loss(batch)
                policy_loss, entropies = self.policy_loss(batch)

                self.update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)
                self.update_params(self.critic_q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
                self.update_params(self.critic_q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

                # Tune entropy values
                entropy_loss = self.entropy_loss(entropies)
                self.update_params(self.alpha_optim, None, entropy_loss)
                self.alpha = self.log_alpha.exp()

                # Bookkeeping
                self.writer.add_scalar('loss/alpha', entropy_loss.detach().item(), self.step_count)
                self.writer.add_scalar('loss/Q1', q1_loss.detach().item(), self.step_count)
                self.writer.add_scalar('loss/Q2', q1_loss.detach().item(), self.step_count)
                self.writer.add_scalar('loss/policy', policy_loss.detach().item(), self.step_count)
                self.writer.add_scalar('stats/alpha', self.alpha.detach().item(), self.step_count)
                self.writer.add_scalar('stats/mean_Q1', mean_q1, self.step_count)
                self.writer.add_scalar('stats/mean_Q2', mean_q2, self.step_count)
                self.writer.add_scalar('stats/entropy', entropies.detach().mean().item(), self.step_count)

            if episode_steps >= self.env._max_episode_steps or done:
                break

            state = next_state

        self.writer.add_scalar('Episode_rewards', episode_reward, epoch)
        self.writer.add_scalar('Episode_steps', episode_steps, epoch)

        print(f"Epoch: {epoch:<4} || Episode_steps: {episode_steps:<4} || Episode_rewards: {episode_reward:<4}")

    def act(self, epoch, state, evaluate=False):

        if evaluate:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, action = self.policy(state)
        elif epoch < self.exploration_epochs:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _ = self.policy(state)

        return action.cpu().numpy().reshape(-1)

    def save_models(self, epoch):
        self.policy.save(os.path.join(self.save_dir, f'{epoch}_policy.pth'))
        self.critic.save(os.path.join(self.save_dir, f'{epoch}_critic.pth'))
        self.critic_target.save(os.path.join(self.save_dir, f'{epoch}_critic_target.pth'))

    def entropy_loss(self, entropies):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach())
        return entropy_loss

    def policy_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy))
        return policy_loss, entropy

    def current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma * next_q

        return target_q

    def critic_loss(self, batch):
        curr_q1, curr_q2 = self.current_q(*batch)
        target_q = self.target_q(*batch)

        # TD error
        td_error = torch.abs(curr_q1.detach() - target_q)

        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))
        return q1_loss, q2_loss, td_error, mean_q1, mean_q2

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for episode in range(episodes):
            state, info = self.env.reset()
            episode_reward = 0.
            done = False

            step = 0
            while not done and step < self.env._max_episode_steps:
                if self.render:
                    self.env.render()
                action = self.act(episode, state, True)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                step += 1

            returns[episode] = episode_reward

        mean_return = np.mean(returns)

        self.writer.add_scalar('reward/eval', mean_return, self.eval_step)
        self.eval_step += 1
        print('-' * 60)
        print(f'Num steps: {self.eval_step:<5} || reward: {mean_return:<5.1f}')
        print('-' * 60)

    def make_video(self, path, n=1):
        print(n)
        env = RecordVideo(self.env, path)

        for i in range(n):
            rewards = 0
            steps = 0
            done = False
            observation, _ = env.reset()
            env.start_video_recorder()

            while True :
                action = self.act(i, observation, True)
                observation, reward, done, _, _ = env.step(action)
                steps += 1
                rewards += reward

                if steps >= 1000 or done:
                    break

            print("-" * 80)
            print("Episode steps: {} || Rewards {}: ".format(steps, rewards))
            print("-" * 80)

        env.close()

    @staticmethod
    def soft_update(target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

    @staticmethod
    def hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def grad_false(network):
        for param in network.parameters():
            param.requires_grad = False

    @staticmethod
    def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if grad_clip is not None:
            for p in network.modules():
                torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
        optim.step()
