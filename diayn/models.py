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
    def __init__(self, env , nskills):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod()+nskills, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

class FeatureNetwork(nn.Module):
    def __init__(self, env , sf_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, sf_dim),
        )

    def forward(self, x):
        basis =  self.network(x)
        basis_normalised =  F.normalize(basis, p=2, dim=-1)
        return basis , basis_normalised


class SFNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, sf_dim=32, hidden_sizes=(120, 84)):
        super(SFNetwork, self).__init__()
        self.input_dim = state_dim + action_dim
        self.sf_dim = sf_dim

        self.l1 = nn.Linear(self.input_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], sf_dim)


    def argforward(self, state, action, weights , task):
        x = torch.cat([state, action], dim=-1)
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        
        q_pred = torch.einsum("bi,bi->b", task, x)
        return q_pred


