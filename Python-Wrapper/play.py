from jetbotSim import Robot, JeltBot
from ppo import Actor
from module import CNN
from network import Dueling_Net
import torch

def discrete_action_encoder(action):
    if action == 0:
        return [0.35, 0.35]
    elif action == 1:
        return [0.12, 0.05]
    elif action == 2:
        return [0.05, 0.12]

class Agent():
    def __init__(self, env, model_dir='best_ppo_3e-1'):
        device = torch.device('cpu')

        obs = env.reset()
        self.actor = Actor(CNN(obs), 256, 256, 3)
        self.actor.load_state_dict(torch.load(model_dir, map_location=device)['actor'])
        torch.save(self.actor, 'PPO')

    def select_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)

        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()
            #action = torch.argmax(dist.probs)
            action = action.item()

        return action

class DQNarg():
    def __init__(self):
        self.hidden_dim = 256
        self.state_dim = (3,64,64)
        self.action_dim = 3
        self.use_noisy = False#True

class DQNAgent():
    def __init__(self, model_dir):
        args = DQNarg()
        device = torch.device('cpu')

        self.net = Dueling_Net(args)
        self.net.eval()
        self.net.load_state_dict(torch.load(model_dir, map_location=device))

    def select_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)

        with torch.no_grad():
            q = self.net(state)
            action = q.argmax(dim=-1).item()

        return action

if __name__ == '__main__':
    model_dir = 'BEST_PPO_35e-2_12e-2_5e-2'

    robot = Robot()
    env = JeltBot(robot, action_encoder=discrete_action_encoder, skipping_time=0, max_step=200, n_frame_stack=4)
    model = Agent(env, model_dir)#DQNAgent(model_dir)#

    for _ in range(10):
        terminated = truncated = False
        total_reward = steps = 0
        observation = env.reset()
        
        while not terminated and not truncated:
            action = model.select_action(observation)
            next_observation, reward, terminated, truncated = env.step(action)
            
            total_reward += reward
            steps += 1
            
            observation = next_observation
        print('Ep.Reward: %.3f\tSteps: %d'%(total_reward, steps))

    


    