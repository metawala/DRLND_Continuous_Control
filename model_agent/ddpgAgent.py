
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import logging

from model_agent.models import Actor, Critic
from collections import namedtuple, deque



class ReplayBuffer():
    def __init__(self, actionSize, bufferSize, batchSize, seed):
        self.actionSize = actionSize
        self.memory = deque(maxlen=bufferSize)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])
        self.seed = random.seed(seed)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, reward, nextState, done):
        exp = self.experience(state, action, reward, nextState, done)
        self.memory.append(exp)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batchSize)

        states     = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(self.device)
        actions    = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).float().to(self.device)
        rewards    = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(self.device)
        nextStates = torch.from_numpy(np.vstack([exp.nextState for exp in experiences if exp is not None])).float().to(self.device)
        dones      = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, nextStates, dones)

    def __len__(self):
        return len(self.memory)

class OUNoise():
    def __init__(self, size, seed, mu = 0.0, theta = 0.15, sigma = 0.2):
        self.mu     = mu * np.ones(size)
        self.signma = sigma
        self.theta  = theta
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.signma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state

class Agent():
    def __init__(self, stateSize, actionSize, randomSeed):
        self.stateSize  = stateSize
        self.actionSize = actionSize
        self.seed = random.seed(randomSeed)
        self.bufferSize = int(10e6)
        self.batchSize  = 1024

        self.logger = logging.getLogger(self.__class__.__name__)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Actor local and target network params
        self.actorLocal     = Actor(stateSize, actionSize, randomSeed).to(self.device)
        self.actorTarget    = Actor(stateSize, actionSize, randomSeed).to(self.device)
        self.actorOptimizer = optim.Adam(self.actorLocal.parameters(), lr = 5e-4)
        
        # Critic local and target network params
        self.criticLocal     = Critic(stateSize, actionSize, randomSeed).to(self.device)
        self.criticTarget    = Critic(stateSize, actionSize, randomSeed).to(self.device)
        self.criticOptimizer = optim.Adam(self.criticLocal.parameters(), lr = 1e-3, weight_decay = 0)

        self.noise = OUNoise(actionSize, randomSeed)

        self.memory = ReplayBuffer(actionSize, self.bufferSize, self.batchSize, randomSeed)

    def step(self, state, action, reward, nextState, done, timestep):
        #self.memory.add(state, action, reward, nextState, done)
        for i in range(20):
            self.memory.add(state[i], action[i], reward[i], nextState[i], done[i])

        if timestep % 20 == 0:
            if len(self.memory) > self.batchSize:
                for i in range(10):
                    experiences = self.memory.sample()
                    self.learn(experiences, 0.99)
        
    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise = True):
        '''
        state = torch.from_numpy(state).float().to(self.device)
        self.actorLocal.eval()
        with torch.no_grad():
            actionValues = self.actorLocal.forward(state)

        self.actorLocal.train()
        actionValues += self.noise.sample()
        actionValues = np.clip(actionValues, -1, 1)

        return actionValues
        '''

        state = torch.from_numpy(state).float().to(self.device)
        self.actorLocal.eval()
        with torch.no_grad():
            action = self.actorLocal(state).cpu().data.numpy()
        self.actorLocal.train()
        if add_noise:
            for i in range(20):
                action[i] += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        states, actions, rewards, nextStates, dones = experiences

        actionsPred = self.actorLocal.forward(states)
        actionsLoss = -self.criticLocal.forward(states, actionsPred).mean()
        self.actorOptimizer.zero_grad()
        actionsLoss.backward()
        self.actorOptimizer.step()

        actionsNext = self.actorTarget.forward(nextStates)
        QTargetNext = self.criticTarget.forward(nextStates, actionsNext)
        QTargets = rewards + (gamma * QTargetNext * (1 - dones))
        QExpected = self.criticLocal.forward(states, actions)

        criticLoss = F.mse_loss(QExpected, QTargets)
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        self.softUpdate(self.actorLocal, self.actorTarget, 1e-3)
        self.softUpdate(self.criticLocal, self.criticTarget, 1e-3)

    def softUpdate(self, localModel, targetModel, tau):
        for targetParam, localParam in zip(targetModel.parameters(), localModel.parameters()):
            targetParam.data.copy_(tau * localParam.data + (1.0 - tau) * targetParam.data)