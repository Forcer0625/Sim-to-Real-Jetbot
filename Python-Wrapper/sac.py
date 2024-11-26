import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from utilts import *
from module import *
from torch.utils.tensorboard import SummaryWriter
from time import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC():
    def __init__(self, config):
        self.env = config['env']
        self.device = config['device']
        self.n_actions = config['n_actions']

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.lr = config['lr']
        self.scale = config['reward_scale']
        
        obs = self.env.reset()
        self.buffer_size = config['buffer_size']
        self.memory = ReplayBuffer(self.buffer_size,
                                   obs.shape,
                                   self.n_actions)
        self.batch_size = config['batch_size']

        self.actor = Actor(CNN(obs), self.n_actions, device=self.device, lr=self.lr)
        self.critic1=Critic(CNN(obs), self.n_actions, device=self.device, lr=self.lr)
        self.critic2=Critic(CNN(obs), self.n_actions, device=self.device, lr=self.lr)
        self.q_net = QNetwork(CNN(obs), device=self.device, lr=self.lr)
        self.target = QNetwork(CNN(obs), device=self.device, lr=self.lr)
        self.target.load_state_dict(self.q_net.state_dict())
    
    def select_action(self, obs:np.ndarray):
        batch_stack_frame = np.expand_dims(obs, axis=0)
        feature = torch.as_tensor(batch_stack_frame, dtype=torch.float32).to(self.device)
        actions, _ = self.actor.sample(feature, False)
        return actions.cpu().detach().numpy()[0]
    
    def store(self, obs, action, reward, obs_, done):
        self.memory.store(obs, action, reward,\
                          obs_, done)

    def sync(self, tau=None):
        if tau is None:
            tau = self.tau
        target_params = self.target.named_parameters()
        q_net_params = self.q_net.named_parameters()

        target_state_dict = dict(target_params)
        q_net_state_dict = dict(q_net_params)

        for name in q_net_state_dict:
            q_net_state_dict[name] = tau*q_net_state_dict[name].clone() + (1-tau)*target_state_dict[name].clone()
        
        self.target.load_state_dict(q_net_state_dict)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return np.nan
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done).to(self.device)

        value = self.q_net(state).view(-1)
        value_= self.target(next_state).view(-1)
        value_[done] = .0

        actions, log_probs = self.actor.sample(state, False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        cirtic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)
        
        self.q_net.optimizer.zero_grad()
        value_target = cirtic_value - log_probs
        value_loss = .5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.q_net.optimizer.step()

        actions, log_probs = self.actor.sample(state, False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        cirtic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = torch.mean(log_probs - cirtic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic1.forward(state, action).view(-1)
        q2_old_policy = self.critic2.forward(state, action).view(-1)
        critic1_loss = .5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = .5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.sync()
    
    def save_policy(self, path='actor'):
        torch.save(self.actor.state_dict(), path)

    def load_policy(self, path='actor'):
        self.actor.load_state_dict(torch.load(path))

if __name__ == "__main__":
    print(DEVICE)
    env = None
    config = {
        'env':env,
        'gamma':0.99,
        'lr': 3e-4,
        'tau':0.01, # more is harder
        'step_size':5,
        'reward_scale':5,
        'n_frame_stack':4,
        'batch_size':12,
        'buffer_size':60000,
    }
    print_every = 1
    save_every = 200
    n_frame_skipping = 4
    agent = SAC(config)
    mean_reward = []
    mean_update_time = []
    mean_event_time = []
    writer = SummaryWriter('./runs')

    TOTAL_EPISODE = int(1e7)
    for episode in range(TOTAL_EPISODE):
        obs = env.reset()
        agent.frame_buffer.clear()
        total_reward = 0
        done = False
        ep_time = time()
        while not done:
            action = agent.select_action(obs)
            skipping_reward = 0
            encoded_action = action#action_encode(action)
            #t = time()
            #print(encoded_action)
            for _ in range(n_frame_skipping):
                #env.render()
                obs_, reward, done, info = env.step(encoded_action)
                skipping_reward += reward
                if done:
                    break
            #event_time = time() - t
            total_reward += skipping_reward
            #t = time()
            agent.store(obs, action, skipping_reward/n_frame_skipping, obs_, done)
            agent.update()
            #update_time = time() - t
            obs = obs_
            #mean_update_time.append(update_time)
            #mean_event_time.append(event_time)
        ep_time = time() - ep_time
        mean_reward.append(total_reward)
        mean_event_time = []
        mean_update_time = []
        if episode % save_every == 0:
            agent.save_policy('112062546_hw3_data_episode%d'%(episode))
        if episode % print_every == 0:
            print('%d Episode Time:%.2f, Ep.Reward:%.2f, Ave.Reward:%.2f'%\
                    (episode, ep_time, total_reward, np.mean(mean_reward[-100:])))
            if len(mean_reward) > 500:
                mean_reward = mean_reward[300:]
        writer.add_scalar('Train/Episode Reward', total_reward, episode)
        writer.add_scalar('Train/Ave. Reward', np.mean(mean_reward[-100:]), episode)
    agent.save_policy()