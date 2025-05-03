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
        self.hidden1 = nn.Linear(n_states, 120)
        init_weight(self.hidden1)

        self.hidden2 = nn.Linear(120, 32)
        init_weight(self.hidden2)

        self.q = nn.Linear(32, n_skills)
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
            # nn.Linear(84, 84),
            # nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

import torch
import torch.nn as nn
import numpy as np

class FeatureNetwork(nn.Module):
    def __init__(self, env, sf_dim, dropout_p=0.1):  # ← allow dropout probability
        super().__init__()
        input_dim = np.prod(env.observation_space.shape)

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, sf_dim)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, task):
        """
        Args:
            x: tensor of shape (batch_size, state_dim)
            task: tensor of shape (batch_size, sf_dim)
        Returns:
            predicted reward: tensor of shape (batch_size,)
        """
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x1))
        x_res = x1 + x2

        x3 = self.activation(self.fc3(x_res))
        x3 = self.dropout(x3)  # ← dropout added here
        phi = self.fc4(x3)

        q_pred = torch.einsum("bi,bi->b", phi, task)
        return q_pred
    
    def forward2(self, x):
        """
        Args:
            x: tensor of shape (batch_size, state_dim)
            task: tensor of shape (batch_size, sf_dim)
        Returns:
            predicted reward: tensor of shape (batch_size,)
        """
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x1))
        x_res = x1 + x2

        x3 = self.activation(self.fc3(x_res))
        # x3 = self.dropout(x3)  # ← dropout added here
        phi = self.fc4(x3)

        return phi

class SFNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, sf_dim=32):
        super(SFNetwork, self).__init__()
        self.input_dim = state_dim + action_dim
        self.sf_dim = sf_dim

        self.l1 = nn.Linear(self.input_dim, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, sf_dim)


    def argforward(self, state, action, weights , task):
        x = torch.cat([state, action], dim=-1)
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        task = task.unsqueeze(0).expand(x.size(0), -1)
        q_pred = torch.einsum("bi,bi->b", task, x)
        return q_pred
    
    def forward(self, state, action, task):
        """
        Plain forward with the *current* parameters.
        This is what wandb will capture in its graph.
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # broadcast task to batch:
        task = task.unsqueeze(0).expand(x.size(0), -1)
        return torch.einsum("bi,bi->b", task, x)


class QNetworkMaml(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            # nn.Linear(84, 84),
            # nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)
    
    def argforward(self, state, weights):
        x = F.linear(state, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        return x