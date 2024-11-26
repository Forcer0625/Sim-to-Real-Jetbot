import sys
sys.path.append('./Python-Wrapper/jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import config
from robot import Robot
from utilts import FrameStack, downsample, resize_transpose

class Env():
    def __init__(self):    
        self.ws = None
        self.wst = None
        self._connect_server(config.ip, config.actor)
        self.buffer = None
        self.on_change = False

    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp("ws://%s/%s/camera/subscribe"%(ip, actor), on_message = lambda ws,msg: self._on_message_env(ws, msg))
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        time.sleep(1)   #wait for connect
    
    def _on_message_env(self, ws, msg):
        self.buffer = msg
        self.on_change = True
        
    def run(self, execute):
        #print("\n[Start Observation]")
        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                reward = int.from_bytes(self.buffer[:4], 'little')
                done = bool.from_bytes(self.buffer[4:5], 'little')
                execute({"img":img.copy(), "reward":reward, "done":done})
                self.on_change = False

class JeltBot(Env):
    def __init__(self, robot:Robot, action_encoder=None, n_frame_stack=4, skipping_time=0.02, max_step=25):
        super().__init__()
        self.n_frame_stack = n_frame_stack
        self.skipping_time = skipping_time
        self.robot = robot
        if self.n_frame_stack > 1:
            self.frame_buffer = FrameStack(n_frame_stack)
        self.action_encoder = action_encoder
        self.max_step = max_step
        self.step_counter = 0

    def reset(self):
        self.robot.set_motor(0.0, 0.0)
        self.robot.reset()
        time.sleep(0.1)
        self.step_counter = 0
        self.prev_reward = 0
        
        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.on_change = False
                break
        self.frame = img
        if self.n_frame_stack > 1:
            self.frame_buffer.clear()
            self.frame_buffer.push(downsample(img))
            obs = self.frame_buffer.get()
        else:
            obs = resize_transpose(img)
        return obs

    def step(self, action):
        '''return next_obs, reward, is_terminated, is_truncated'''
        

        if self.action_encoder is not None:
            action = self.action_encoder(action)

        self.robot.set_motor(action[0], action[1])
        if self.skipping_time > 0.0:
            time.sleep(self.skipping_time)

        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                reward = int.from_bytes(self.buffer[:4], 'little')
                done = bool.from_bytes(self.buffer[4:5], 'little')
                self.on_change = False
                break

        self.frame = img
        if self.n_frame_stack > 1:
            self.frame_buffer.push(downsample(img))
            obs = self.frame_buffer.get()
        else:
            obs = resize_transpose(img)
        
        #print('red: %.2f\tblue: %.2f\tgreen: %.2f'%(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])), end='\r')
        
        if reward > 0:
            self.step_counter = 0
        else:
            reward = 1
            if not (action[0]==0.0 and action[1]==0.0):    
                self.step_counter += 1
                if self.step_counter >= self.max_step*0.2:
                    reward = -1#-0.5*self.step_counter
                elif self.step_counter >= self.max_step*0.1:
                    reward = 0
        truncated = done#self.step_counter >= self.max_step
        done = self.step_counter >= self.max_step
        #self.prev_reward = reward
        #reward -= 0.1#1.0/self.max_step
        #print('%.3f'%(reward), end="\r")
        #self.robot.set_motor(0.0,0.0)
        #self.frame_buffer.clear()
        return obs, reward, done, truncated
    
    def render(self):
        cv2.imshow('JeltBot Camara', self.frame)
        cv2.waitKey(32)
        

        