try: 
    import gymnasium as gym 
    gm = True
except: 
    pg = True 
    
from MissileEnv import MissileEnv 

# Register the environment, so we can use gymnasium.make(), allowing us to make the most of the library 
env_id = "gym_env:MissileEnv-v2"
gym.register(
    id=env_id, 
    entry_point=MissileEnv
)
print("Successful Register")
gym.pprint_registry()