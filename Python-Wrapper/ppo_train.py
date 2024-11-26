from jetbotSim import Robot, Env, JeltBot
import numpy as np
import cv2
from ppo import PPO
from utilts import *
from torch.utils.tensorboard import SummaryWriter
import time

class Logger():
    def __init__(self, path='./runs'):
        self.writer = SummaryWriter(path)

    def log(self, step, ep_reward, avg_reward):
        self.writer.add_scalar('Train/Episode Reward', ep_reward, step)
        self.writer.add_scalar('Train/Ave. Reward', avg_reward, step)

def discrete_action_encoder(action):
    #v = 0.5
    if action == 0:
        return [0.35, 0.35]
    elif action == 1:
        return [0.12, 0.05]
    elif action == 2:
        return [0.05, 0.12]
    # elif action == 3:
    #     return [0.0, 0.0]
    # elif action == 4:
    #     return [-0.2, -0.2]
    
if __name__ == '__main__':
    robot = Robot()
    env = JeltBot(robot, action_encoder=discrete_action_encoder, skipping_time=0.1, max_step=40, n_frame_stack=1)
    config = {
        'n_actions':3,
        'gamma':0.99,
        'gae_lambda':0.95,
        'step_size':1024,
        'clip_size':0.2,
        'epoch_size':4,
        'mem_size':4096,
        'entropy_coff':0.05,
        'lr':3e-4,
        'layer_dims':(256, 256),
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    agent = PPO(env, config)
    timestamp = time.asctime(time.localtime(time.time())).replace(':', '-')
    
    logger = Logger('./runs/'+'SingleFramePPO'+timestamp)
    TOTAL_TIMESTEPS = int(1e6)
    agent.load('SingleFramePPOMon Jun  3 05-35-17 2024')
    try:
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, render=True, logger=logger)
    finally:
        agent.save('SingleFramePPO'+timestamp)

    exit()
    for i in range(100):
        score = 0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            with torch.no_grad():
                action, prob, val = agent.select_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            obs = obs_
            score+=reward
        env.close()