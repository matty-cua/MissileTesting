import gymnasium as gym 
from MissileEnv import MissileEnv 

# Register the environment, so we can use gymnasium.make(), allowing us to make the most of the library 
env_id = "gymnasium_env/MissileEnv-v1"
gym.register(
    id=env_id, 
    entry_point=MissileEnv
)