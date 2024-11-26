from jetbotSim import Robot, Env
import numpy as np
import cv2
import matplotlib.pyplot as plt

def step(action):
    global robot
    if action == 0:
        robot.set_motor(0.5, 0.5)
    elif action == 1:
        robot.set_motor(0.2, 0.)
    elif action == 2:
        robot.set_motor(0., 0.2)

def execute(obs):
    # Visualize
    global frames
    frames += 1
    img = obs["img"]
    #cv2.imshow(f'test_img.png', img)
    #cv2.waitKey(32)
    reward = obs['reward']
    done = obs['done']
    if frames < 100:
        step(0)
    else:
        frames = 0
        robot.reset()
    print('\rframes:%d, reward:%.2f, done:%d ' % (frames, reward, done), end = "")

frames = 0
robot = Robot()
env = Env()
env.run(execute)