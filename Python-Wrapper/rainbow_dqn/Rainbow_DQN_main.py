import torch
import numpy as np
from replay_buffer import *
from jetbotSim import Robot, Env, JeltBot
from rainbow_dqn import DQN
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

def discrete_action_encoder(action):
    if action == 0:
        return [0.5, 0.5]
    elif action == 1:
        return [0.2, 0.0]
    elif action == 2:
        return [0.2, 0.2]

class Logger():
    def __init__(self, path='./runs'):
        self.writer = SummaryWriter(path)

    def log(self, step, ep_reward):
        self.writer.add_scalar('Train/Episode Reward', ep_reward, step)

class Runner:
    def __init__(self, args):
        self.args = args

        robot = Robot()
        env = JeltBot(robot, action_encoder=discrete_action_encoder, skipping_time=0, max_step=30)
        self.env = env
        obs = env.reset()

        self.args.state_dim = obs.shape
        self.args.action_dim = 3

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, render, logger:Logger):
        while self.total_steps < self.args.max_train_steps:
            state = self.env.reset()
            done = False
            episode_steps = 0
            total_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done, truncated = self.env.step(action)
                self.env.robot.set_motor(0.0, 0.0)
                episode_steps += 1
                self.total_steps += 1
                total_reward += reward

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # truncate means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                # if done and episode_steps != self.args.episode_limit:
                #     if self.env_name == 'LunarLander-v2':
                #         if reward <= -100: reward = -1  # good for LunarLander
                #     truncate = True
                # else:
                #     truncate = False

                self.replay_buffer.store_transition(state, action, reward, next_state, False, done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

            if render:
                print('total_steps ', self.total_steps, 'ep.steps ', episode_steps, 'reward %.2f'%total_reward)
            
            if logger is not None:
                logger.log(self.total_steps, total_reward)
    
    def save(self, path):
        torch.save(self.agent.net.state_dict(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(5e5), help=" Maximum number of training steps")

    parser.add_argument("--buffer_capacity", type=int, default=int(5e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.8, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(2e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    runner = Runner(args)

    timestamp = time.asctime(time.localtime(time.time())).replace(':', '-')
    logger = Logger('./runs/'+'DQN'+timestamp)

    try:
        runner.run(render=True, logger=logger)
    finally:
        runner.save('DQN'+timestamp)
    # env_names = ['CartPole-v1', 'LunarLander-v2']
    # env_index = 1
    # for seed in [0, 10, 100]:
    #     runner = Runner(args=args, env_name=env_names[env_index], number=1, seed=seed)
    #     runner.run()