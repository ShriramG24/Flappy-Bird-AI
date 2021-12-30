import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random, math
import gym
import numpy as np


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LinearDQN(nn.Module):
    def __init__(self, height, width):
        super(LinearDQN, self).__init__()
        self.layer1 = nn.Linear(in_features=height*width*3, out_features=24)
        self.layer2 = nn.Linear(in_features=24, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=2)

    def forward(self, img):
        img = img.flatten(start_dim=1)
        img = F.relu(self.layer1(img))
        img = F.relu(self.layer2(img))
        img = self.output(img)
        return img


class Agent:
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

    def get_exploration_rate(self, start, end, decay):
        return end + (start - end) * math.exp(-1. * self.current_step * decay)

    def select_action(self, obs, policy_net):
        rate = self.get_exploration_rate(0, 0, 0)
        self.current_step += 1
        if rate > random.random():
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                return policy_net(obs).argmax(dim=1).item()






