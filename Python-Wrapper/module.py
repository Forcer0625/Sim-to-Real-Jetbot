import torch
import torch.nn as nn
import numpy as np
from utilts import downsample, crop, resize, grayscale
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

class CNN(nn.Module):
    def __init__(self, obs):
        super().__init__()
        self.in_channels = obs.shape[0]
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        batch_stack_frame = np.expand_dims(obs, axis=0) # [C,H,W] -> [B,C,H,W]

        with torch.no_grad():
            batch_stack_frame = torch.as_tensor(batch_stack_frame, dtype=torch.float32)
            n_flatten = self.model(batch_stack_frame).shape[1]

        self.out_dims = n_flatten

    def forward(self, x):
        x = x/255.0
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, feature_extracter:CNN, n_actions, device, lr=1e-4):
        super(Critic, self).__init__()
        #self.in_dims = in_dims
        self.n_actions = n_actions
        self.feature_extracter = feature_extracter
        self.fc1 = nn.Linear(feature_extracter.out_dims+n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state, action):
        feature = self.feature_extracter(state)
        action_value = self.fc1(torch.cat([feature, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        return self.q(action_value)

class QNetwork(nn.Module):
    def __init__(self, feature_extracter:CNN, device, lr=1e-4):
        super(QNetwork, self).__init__()
        
        self.feature_extracter = feature_extracter
        self.fc1 = nn.Linear(feature_extracter.out_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        feature = self.feature_extracter(x)
        state_value = self.fc1(feature)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        return self.value(state_value)

class Actor(nn.Module):
    def __init__(self, feature_extracter:CNN, n_actions, device, lr=1e-4):
        super(Actor, self).__init__()
        self.reparam_noise = 1e-6

        self.feature_extracter = feature_extracter
        self.fc1 = nn.Linear(feature_extracter.out_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, n_actions)
        self.sigma = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        feature = self.feature_extracter(state)
        action_prob = self.fc1(feature)
        action_prob = F.relu(action_prob)
        action_prob = self.fc2(action_prob)
        action_prob = F.relu(action_prob)

        mean = self.mean(action_prob)
        sigma = torch.clamp(self.sigma(action_prob), min=self.reparam_noise, max=1)

        return mean, sigma
    
    def sample(self, state, reparameterize=True):
        mean, sigma = self.forward(state)
        probs = Normal(mean, sigma)

        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        
        action = torch.tanh(actions)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs