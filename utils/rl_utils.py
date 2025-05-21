from collections import namedtuple
from collections import deque
import random
import math
import torch
import ipdb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
                        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):    
        #print(len(self.memory))
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])
        return states, actions, rewards, next_states, dones
    
    def syn_sample(self, sample_positions):    
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for pos in sample_positions: 
            states.append(self.memory[pos][0])
            actions.append(self.memory[pos][1])
            rewards.append(self.memory[pos][2])
            next_states.append(self.memory[pos][3])
            dones.append(self.memory[pos][4])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


def e_greedy(voter_algo, obs, training_steps, environment, voter_name, test=False, cap=False, ballot='approval'):
    """
    This does the epsilon greedy bit
    """
    if test:
        with torch.no_grad():
            actions = voter_algo.get_action(obs.flatten())
            return actions

    EPS_START = 0.5
    EPS_END = 0.010
    EPS_DECAY = 80

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * training_steps / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            actions = voter_algo.get_action(obs.flatten())
            return actions
    else:
        if cap:
            actions = torch.zeros(voter_algo.q.ac_dim, dtype=int)
            for _ in range(int(len(actions/2))):
                i = random.randint(0,len(actions)-1)
                actions[i]=1
                return actions
            
        ass = environment.action_space(environment.possible_agents[0])
        actions = torch.stack([
            torch.tensor(ass.sample(), device=device, dtype=torch.long).type(torch.int64)
            for i in range(voter_algo.q.ac_dim)
            ])
        return actions
    
def category_dict_to_name(category_dict):
    present_categories = []
    for category, presence in category_dict.items():
        if presence == 1:
            present_categories.append(category)
    return present_categories
