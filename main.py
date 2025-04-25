# imports 
# from tqdm import tqdm
import matplotlib.pyplot as plt 

from PathGen import PathGenerator

# risky pygame import (used for rendering) 
try: 
    import pygame 
    pg = True
except: 
    print("WARNING: COULD NOT IMPORT PYGAME. WILL NOT RENDER.")
    pg = False

# risky gym import 
try: 
    import gymnasium as gym 
    gm = True
except: 
    print("WARNING: COULD NOT IMPORT GYMNASIUM. TRAINING AND CLASSES MAY FAIL")
    gm = False; 

# custom imports 
from Tools import * 
from MissileEnv import * 

pathGen = PathGenerator()
(px, py) = pathGen.get_path(100)

# define hyper parameters 
epochs = 1

# Initialize the environment 
env = MissileEnv(); 
env.render_mode = "human"
plot_path = False

# env_id = "gymnasium_env/MissileEnv-v1"
# env = gym.make(
#     env_id, 
#     render_mode="human", 
# )

# Training loop
for episode in (range(epochs)): 
    print(f"Epoch {episode}")
    env.reset()
    if plot_path: 
        plt.plot(env.target_x, env.target_y)
        plt.show()
    frms = 0; 
    done = False 
    while not done: 
        next_obs, reward, terminated, truncated, info = env.step()
        done = terminated or truncated
        frms += 1
        pd = env.cam_pos(env.target.position)
        print("=====")
        print(f"{env.target.position.x:.2f}, {env.target.position.y:.2f}")
        print(f"{pd[0]:.2f}, {pd[1]:.2f}")
        # print(f"Frames: {frms}")