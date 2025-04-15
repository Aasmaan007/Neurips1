import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)
    layer.bias.data.zero_()

class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills):
        super(Discriminator, self).__init__()
        self.input_dim = n_states
        self.hidden1 = nn.Linear(n_states, 64)
        init_weight(self.hidden1)

        self.hidden2 = nn.Linear(64, 64)
        init_weight(self.hidden2)

        # self.hidden3 = nn.Linear(120, 64)
        # init_weight(self.hidden3)

        # self.hidden4 = nn.Linear(64, 8)
        # init_weight(self.hidden4)

        self.q = nn.Linear(64, n_skills)
        init_weight(self.q, initializer="xavier uniform")

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        logits = self.q(x)
        return logits

class QNetwork(nn.Module):
    def __init__(self, env , nskills):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod()+nskills, 120),
            nn.ReLU(),
            # nn.Linear(120, 120),
            # nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)
