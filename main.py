# imports 
# from tqdm import tqdm

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

# define hyper parameters 
epochs = 4

# Initialize the environment 
env = MissileEnv(); 
env.render_mode = "human"
# env_id = "gymnasium_env/MissileEnv-v1"
# env = gym.make(
#     env_id, 
#     render_mode="human", 
# )

# Training loop
for episode in (range(epochs)): 
    print(f"Epoch {episode}")
    env.reset()
    frms = 0; 
    done = False 
    while not done: 
        next_obs, reward, terminated, truncated, info = env.step()
        done = terminated or truncated
        frms += 1
        print(f"Frames: {frms}")