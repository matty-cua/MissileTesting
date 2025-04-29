print("Misc Imports...") 
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from collections import namedtuple, deque 
import random 
from pathlib import Path 
import numpy as np

print("Torch imports...")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("Custom imports...")
from MissileEnv import MissileEnv

print("Finished imports.")

# Psuedo enum (pytorch tutorial) 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Class to hold data from replay 
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Define a multi layer densely connected network for DQN policy estimation 
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_RNN(nn.Module): 

    def __init__(self, n_observations, n_actions):
        super(DQN_RNN, self).__init__()
        self.rnn1 = nn.RNN(n_observations, 128)
        self.rnn2 = nn.RNN(128, 128)
        self.dense = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): 
        output1, hidden1 = self.rnn1(x)
        output2, hidden2 = self.rnn2(hidden1)
        action = self.dense(hidden2)
        action = self.softmax(action)
        return action 
    
def select_action(state):
    """
    Training policy, this is epsilon greedy 
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

def episode_plot(data, ylbl, show_result=False):
    plt.figure(1)
    pdata = torch.tensor(data, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel(ylbl)
    plt.plot(pdata.numpy())
    plt.grid()
    # Take 100 episode averages and plot them too
    if len(pdata) >= 100:
        means = pdata.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


def optimize_model():
    """
    Training loop 
    """

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    try: 
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    except RuntimeError as e: 
        print("FAILED UPDATE: SKIPPING...")
        print(e)
        return 

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")

# Set up env 
# env = gym.make("CartPole-v1")
# env = gym.make("GymV21Environment-v0")
env = MissileEnv()

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

# Training parameters 
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4

# Set up save path 
save_loc = Path('Output')
save_loc.mkdir(parents=True, exist_ok=True)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
# n_observations = len(state)
n_observations = 5

# Change steps based on hardware acceleration?
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 800
else:
    num_episodes = 50

# Body of the program 
if __name__ == '__main__': 

    # Set up policies 
    print("Setting up policies...")
    model_class = DQN
    policy_net = model_class(n_observations, n_actions).to(device)
    target_net = model_class(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Set up the optimizer and memory (store output from env for training) 
    print("Initialize optimizer and replay memory...")
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    # training step counter + episode duration counter
    steps_done = 0
    episode_durations = []
    average_rewards = []
    epsilons = []
    target_hit = []

    print("Beginning training loop...")
    for i_episode in range(num_episodes):
        print(f"Episode: {i_episode + 1}")
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        reward_total = 0
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            # Update some stats 
            reward_total += reward 

            if terminated:
                next_state = None
                print("    - Hit the Target!")
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                average_rewards.append(reward_total.cpu().numpy()[0] / t + 1)
                cep = eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
                epsilons.append(cep)
                target_hit.append(terminated)
                # plot_durations()
                break

        # Try and print out rewards 
        print(f"    - Duration: {episode_durations[-1]}")
        print(f"    - Reward  : {average_rewards[-1]}")
        print(f"    - Eps     : {cep}")

        if (i_episode + 1) % 50 == 0: 
            try: 
                # Plot and save (can't display it) 
                episode_plot(episode_durations, 'episode')
                plt.gcf().savefig(save_loc / 'DurationPlot.png')
                episode_plot(average_rewards, 'reward')
                plt.gcf().savefig(save_loc / 'RewardPlot.png')
                episode_plot(target_hit, 'target hit')
                plt.gcf().savefig(save_loc / 'TargetHits.png')
                episode_plot(epsilons, 'epsilon')
                plt.gcf().savefig(save_loc / 'EpsilonPlot.png')
            except Exception as e: 
                print('ERROR: Could not show plots: ') 
                print(e)
    
    episode_plot(episode_durations, 'episode')
    plt.gcf().savefig(save_loc / 'DurationPlot.png')
    episode_plot(average_rewards, 'reward')
    plt.gcf().savefig(save_loc / 'RewardPlot.png')
    episode_plot(target_hit, 'target hit')
    plt.gcf().savefig(save_loc / 'TargetHits.png')
    episode_plot(epsilons, 'epsilon')
    plt.gcf().savefig(save_loc / 'EpsilonPlot.png')

    # Save the model 
    torch.save(target_net_state_dict, save_loc / 'ModelWeights.wts')
    torch.save(target_net, save_loc / 'ModelTorch.pkl')

    # Save the training stats (just episode duration for now, should figure out better loss plots like reward and such)
    torch.save(
        {
            'durations': episode_durations, 
            'rewards': average_rewards, 
        }, 
        save_loc / 'ModelStats', 
    )





