import torch
from torch import nn, optim
import torch.nn.functional as func
from collections import deque, namedtuple
import random
import math
import gym
import numpy as np


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.num_exps = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.num_exps % self.capacity] = experience
        self.num_exps += 1

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def can_sample(self, batch_size):
        return batch_size <= self.num_exps


class LinearDQN(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.layer1 = nn.Linear(in_features=width*height, out_features=24)
        self.layer2 = nn.Linear(in_features=24, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = func.relu(self.layer1(t))
        t = func.relu(self.layer2(t))
        t = self.output(t)
        return t


class Agent:
    def __init__(self, num_actions, start, end, decay):
        self.current_step = 0
        self.num_actions = num_actions
        self.start, self.end, self.decay = start, end, decay

    def get_exploration_rate(self):
        return self.end + (self.start - self.end) * math.exp(-1. * self.current_step * self.decay)

    def select_action(self, obs, policy_net):
        rate = self.get_exploration_rate()
        self.current_step += 1
        if rate > random.random():
            return torch.tensor([random.randrange(self.num_actions)]).to("cpu")
        else:
            with torch.no_grad():
                return policy_net(obs).argmax(dim=1).to("cpu")


class QValues:

    @staticmethod
    def get_current(policy, obss, actions):
        return policy(obss).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target, next_obss):
        final_state_locations = next_obss.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_obss[non_final_state_locations]
        batch_size = next_obss.shape[0]
        values = torch.zeros(batch_size).to("cpu")
        values[non_final_state_locations] = target(non_final_states).max(dim=1)[0].detach()
        return values







