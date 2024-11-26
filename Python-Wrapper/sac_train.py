from jetbotSim import Robot, Env, JeltBot
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sac import SAC
from utilts import *
from torch.utils.tensorboard import SummaryWriter
import time

def discrete_action_encoder(action):
    v = 0.5
    if action == 0:
        return [0.0, 0.0]
    elif action == 1:
        return [0.5, 0.5]
    elif action == 2:
        return [0.2, 0.0]
    elif action == 3:
        return [0.0, 0.2]
    elif action == 4:
        return [-0.2, -0.2]

def action_encoder(action):
    #-1.0:left ~ 1.0:right
    # v = 1.0
    # value_l = v*(1.0+action[0])
    # value_r = v*(1.0-action[0])
    # return [value_l, value_r]
    v = 2.0
    encoded_action = [v*(action[0] - (-1.0))/2.0, v*(action[1] - (-1.0))/2.0]
    return encoded_action

if __name__ == '__main__':
    
    robot = Robot()
    env = JeltBot(robot, action_encoder=action_encoder, skipping_time=0)
    config = {
        'env':env,
        'n_actions':2,
        'gamma':0.99,
        'lr': 3e-4,
        'tau':0.01, # more is harder
        'reward_scale':2,
        'init_obs':None,
        'batch_size':256,
        'buffer_size':1000000,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    agent = SAC(config)    
    timestamp = time.asctime(time.localtime(time.time())).replace(':', '-')
    TOTAL_EPISODE = int(1e5)
    writer  = SummaryWriter('./runs/'+'scale'+str(config['reward_scale'])+timestamp)
    print_every = 1
    try:
        steps = 0
        for episode in range(TOTAL_EPISODE):
            obs = env.reset()
            total_reward = 0.0
            terminated = truncated = False
            while (not truncated) and (not terminated):
                #env.render()
                action = agent.select_action(obs)
                obs_, reward, terminated, truncated = env.step(action)

                agent.store(obs, action, reward, obs_, terminated)
                agent.update()
                
                total_reward += reward
                obs = obs_
                steps += 1

            writer.add_scalar('Train/Episode Reward', total_reward, steps)
            # if episode % print_every == 0:
            #     print('Ep.Reward:%.2f'%(total_reward))
    finally:
        env.reset()
        agent.save_policy('SAC-scale'+str(config['reward_scale'])+'_actor_'+timestamp)