import numpy as np
import torch

class ReplayMemory():
    def __init__(self, cfg, device):
        super(ReplayMemory, self).__init__()
        self.capacity = cfg.capacity
        self.state_shape = cfg.state_shape
        self.action_shape = cfg.action_shape
        self.push_count = 0
        self.batch_size = cfg.batch_size
        self.device = device

        self.reset()

    def __len__(self):
        return self.size

    def append(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.states[self.count] = state
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.next_states[self.count] = next_state
        self.dones[self.count] = done

        self.size = min(self.size + 1, self.capacity)
        self.count = (1 + self.count) % self.capacity


    def reset(self):
        self.size = 0
        self.count = 0

        self.states = np.empty((self.capacity, *self.state_shape), dtype=np.float32)
        self.next_states = np.empty((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

    def sample(self):
        indices = np.random.randint(low = 0, high=self.size, size=self.batch_size)

        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def can_provide_sample(self):
        return self.size >= self.batch_size



class PER_ReplayMemory():
    def __init__(self, cfg, device):
        super(PER_ReplayMemory, self).__init__()
        self.capacity = cfg.capacity
        self.state_shape = cfg.state_shape
        self.action_shape = cfg.action_shape
        self.push_count = 0
        self.batch_size = cfg.batch_size
        self.device = device

        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.beta_annealing = cfg.beta_annealing
        self.epsilon = cfg.epsilon

        self.reset()

    def __len__(self):
        return self.size

    def append(self, state, action, reward, next_state, done, error):
        self.priorities[self.count] = self.calculate_priority(error)
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.states[self.count] = state
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.next_states[self.count] = next_state
        self.dones[self.count] = done

        self.size = min(self.size + 1, self.capacity)
        self.count = (1 + self.count) % self.capacity

    def calculate_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(self.calculate_priority(errors), (-1, 1))

    def reset(self):
        self.size = 0
        self.count = 0

        self.states = np.empty((self.capacity, *self.state_shape), dtype=np.float32)
        self.next_states = np.empty((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.priorities = np.empty((self.capacity, 1), dtype=np.float32)

    def sample(self):
        self.beta = min(1-self.epsilon, self.beta + self.beta_annealing)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(self.priorities[:self.size, 0], self.batch_size)
        indices = list(sampler)

        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        p = self.priorities[indices] / np.sum(self.priorities[:self.size])
        weights = (self.size * p) ** (-1 * self.beta)
        weights = torch.FloatTensor(weights / np.max(weights)).to(self.device)

        return (states, actions, rewards, next_states, dones), indices, weights

    def can_provide_sample(self):
        return self.size >= self.batch_size