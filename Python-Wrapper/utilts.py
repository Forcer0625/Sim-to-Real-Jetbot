import numpy as np
import cv2
from collections import deque
from copy import deepcopy
import torch
import torch.nn as nn

def crop(state:np.ndarray):
    '''crop a image from [agent, height, width, channels] -> [H,W,C]'''
    if state.ndim == 3:
        return state[:84,16:80,:]
    return state[0,:84,16:80,:]

def grayscale(state:np.ndarray):
    image = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return image # [H,W]

def resize(state:np.ndarray):
    image = cv2.resize(state, (64, 64), interpolation=cv2.INTER_AREA)
    return image

def downsample(state:np.ndarray):
    return resize(grayscale(state))

def resize_transpose(state:np.ndarray):
    return np.transpose(resize(state), axes=(2,0,1))

class ReplayBuffer():
    def __init__(self, buffer_size:int, state_shape, n_actions):
        self.max_buffer_size = buffer_size
        self.index = 0
        self.buffer_size = 0
        self.states = np.zeros((self.max_buffer_size, *state_shape), dtype=np.uint8)
        self.states_= np.zeros((self.max_buffer_size, *state_shape), dtype=np.uint8)
        self.actions= np.zeros((self.max_buffer_size, n_actions))
        self.rewards= np.zeros(self.max_buffer_size)
        self.dones  = np.zeros(self.max_buffer_size, dtype=bool)

    def store(self, state, action, reward, state_, done):
        self.states [self.index] = state
        self.states_[self.index] = state_
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones  [self.index] = done

        self.index = (self.index+1)%self.max_buffer_size
        self.buffer_size = min(self.buffer_size+1, self.max_buffer_size)

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer_size, batch_size)

        states = self.states [batch]
        states_= self.states_[batch]
        actions= self.actions[batch]
        rewards= self.rewards[batch]
        dones  = self.dones  [batch]

        return states, actions, rewards, states_, dones

    def __len__(self):
        return self.buffer_size

class FrameStack():
    def __init__(self, n_stacks:int=4):
        self.n_stacks = n_stacks
        self.frame_buffer = deque(maxlen=n_stacks)

    def get(self):
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        return stacked_frames

    def push(self, image:np.ndarray):
        self.frame_buffer.append(image)
        while len(self.frame_buffer) < self.n_stacks:
            self.frame_buffer.append(image)

    def render(self):
        pass

    def clear(self):
        self.frame_buffer.clear()
    
    def next_frame(self, image:np.ndarray):
        '''Return stacked frames the next frame'''
        temp = deepcopy(self.frame_buffer)
        temp.append(image)
        return np.stack(temp, axis=0)

