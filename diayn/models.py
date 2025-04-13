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
    def __init__(self, n_states, n_skills, n_hidden_filters=300):
        super(Discriminator, self).__init__()
        self.input_dim = n_states
        self.hidden1 = nn.Linear(n_states, n_hidden_filters)
        init_weight(self.hidden1)

        self.hidden2 = nn.Linear(n_hidden_filters, n_hidden_filters)
        init_weight(self.hidden2)

        self.q = nn.Linear(n_hidden_filters, n_skills)
        init_weight(self.q, initializer="xavier uniform")

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits

class QNetwork(nn.Module):
    def __init__(self, env, n_skills, hidden_dim=300):
        super(QNetwork, self).__init__()
        input_dim = np.prod(env.observation_space.shape) + n_skills
        n_actions = env.action_space.n

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        init_weight(self.fc1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc2)

        self.out = nn.Linear(hidden_dim, n_actions)
        init_weight(self.out, "xavier uniform")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
