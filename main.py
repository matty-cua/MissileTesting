# imports 
# from tqdm import tqdm
import pygame 
import gymnasium as gym 

# custom imports 
from Tools import * 
from MissileEnv import * 

# define hyper parameters 
epochs = 2

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