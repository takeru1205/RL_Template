from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Net
from buffer import ReplayBuffer


class Agent:
    def __init__(
            self,
            env,
            net=None,
            buffer=None,
            gamma=0.95,
            criterion=None,
            optimizer=None,
            cuda=torch.cuda.is_available()
        ):
        self.env = env
        self.buffer = buffer if buffer is not None else ReplayBuffer(
                env.observation_space.shape[0],
                env.action_space.n
            )
        self.gamma = gamma

        self.policy = net if net is not None else Net(
                env.observation_space.shape[0],
                env.action_space.n
            )
        self.target_policy = deepcopy(self.policy)

        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
                self.policy.parameters(),
                lr=0.0001
            )

        self.cuda = cuda
        if self.cuda:
            self.policy = self.criterion.cuda()
            self.target_policy = self.target_policy.cuda()

    def act(self, state):
        return self.env.action_space.sample()

    def store_transition(
            self,
            state,
            action,
            reward,
            next_state,
            done
            ):
        self.buffer.store(state, action, reward, next_state, done)


