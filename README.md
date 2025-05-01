# Missile Testing 

Reinforcement learning experiment for making a target seeking missile. 

To run in interactive mode: $ python Interactive.py 

To train a model: $ python RLManager.py 
* You can tune the hyper parameters after all of the function definitions

To adjust the reward function, go into MissileEnv.py
* The reward is calculated in the step function

Model inputs are a tensor formatted as: 
1. On axis distance to target
2. Off axis distance to target
3. On axis relative velocity to target
4. Off axis relative velocity to target
5. Normalized distance to target (clipped at 300 pixels) (range [0-1] ) 
