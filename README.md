# Missile Testing 
---

Reinforcement learning experiment for making a target seeking missile. 

---
To run in interactive mode: $ python Interactive.py  

---
To train a model in a notebook: ModelTraining.ipynb 
To train a model in a temrinal: $ python RLManager.py 
* You can tune the hyper parameters after all of the function definitions

---
To examine a test episode from a trained model, use the notebook EpisodeExploration.ipynb to load the model, generate and render an episode, and plot summaries of inputs and outputs 

---
To adjust the reward function, go into MissileEnv.py
* The reward is calculated in the method 'step' in the RLManager class 

---
Model inputs are a tensor formatted as: 
1. On axis distance to target
2. Off axis distance to target
3. On axis relative velocity to target
4. Off axis relative velocity to target
5. Normalized distance to target (clipped at 300 pixels) (range [0-1] ) 
