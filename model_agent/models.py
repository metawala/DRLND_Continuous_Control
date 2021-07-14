'''
We will need 2 models. 1 for the Actor and 1 for the Critic.
We will use RELU and TANH activation for the forward path and
Linear fully connected features.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ExtraInit(layer):
    inp = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(inp)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, stateSize, actionSize, seed):
        super(Actor, self).__init__()

        fc1Units = 400
        fc2Units = 300

        self.seed = torch.manual_seed(seed)
        self.BN1 = nn.BatchNorm1d(stateSize)
        self.FC1 = nn.Linear(stateSize, fc1Units)
        self.FC2 = nn.Linear(fc1Units, fc2Units)
        self.FC3 = nn.Linear(fc2Units, actionSize)
        self.initParams()
    
    def initParams(self):
        self.FC1.weight.data.uniform_(*ExtraInit(self.FC1))
        self.FC2.weight.data.uniform_(*ExtraInit(self.FC2))
        self.FC3.weight.data.uniform_(-3e-3, 3e-3)

        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight = nn.init.xavier_uniform(m.weight, gain = 1)
    
    def forward(self, state):
        state = self.BN1(state)
        x = F.relu(self.FC1(state))
        x = F.relu(self.FC2(x))
        x = F.tanh(self.FC3(x))

        return x

class Critic(nn.Module):

    def __init__(self, stateSize, actionSize, seed):
        super(Critic, self).__init__()

        fc1Units = 400
        fc2Units = 300

        self.seed = torch.manual_seed(seed)
        self.BN1 = nn.BatchNorm1d(stateSize)
        self.FC1 = nn.Linear(stateSize, fc1Units)
        self.FC2 = nn.Linear(fc1Units + actionSize, fc2Units)
        self.FC3 = nn.Linear(fc2Units, 1)
        self.initParams()
    
    def initParams(self):
        self.FC1.weight.data.uniform_(*ExtraInit(self.FC1))
        self.FC2.weight.data.uniform_(*ExtraInit(self.FC2))
        self.FC3.weight.data.uniform_(-3e-3, 3e-3)

        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight = nn.init.xavier_uniform(m.weight, gain = 1)

    def forward(self, state, action):
        state = self.BN1(state)
        xs = F.relu(self.FC1(state))
        x = F.relu(self.FC2(torch.cat([xs, action], dim=1)))
        
        return self.FC3(x)