import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 
from collections import namedtuple
from collections import deque
import numpy as np 
import gym 
import random 

import ipdb
NN_SIZE= 276

class BranchingNoDQNetwork(nn.Module):

    def __init__(self, obs, ac_dim, n): 

        super().__init__()

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, 8*n), 
                                   nn.ReLU(),
                                   nn.Linear(8*n,4*n), 
                                   nn.ReLU())

        self.adv_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4*n, n)
            ) 
            for i in range(ac_dim)])
        
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)               
        
        self.model.apply(weights_init)
        for adv_head in self.adv_heads:
            adv_head.apply(weights_init)

    def forward(self, x): 
        x.requires_grad = True
        out = self.model(x)
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)
        q_val = advs - advs.mean(2, keepdim = True )
        return q_val


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class BranchingQN(nn.Module): 
    def __init__(self, obs, ac, n, config, device): 
        super().__init__()
        self.device = device
        self.q = BranchingNoDQNetwork(obs, ac,n ).to(self.device)
        self.update_counter = 0
        self.gamma = 0.2
        self.ac=ac
        
    def get_action(self, x): 
        with torch.no_grad(): 
            out = self.q(x.unsqueeze(0)).squeeze(0)
            action = torch.argmax(out, dim = 1)
        return action

    def update_policy(self, adam, memory, config, batch): 
        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.syn_sample(batch)
        states = torch.tensor(b_states).float().to(self.device)
        actions = torch.tensor(np.array(b_actions)).long().reshape(states.shape[0],-1,1).to(self.device)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1).to(self.device)
        current_q_values = self.q(states).gather(2, actions).squeeze(-1)
        expected_q_vals = torch.cat([rewards]*self.ac,dim=1)

        loss = F.mse_loss(current_q_values, expected_q_vals)
        adam.zero_grad()
        loss.backward(retain_graph = True)

        for p in self.q.parameters(): 
            p.grad.data.clamp_(-1.,1.)

        adam.step()
        self.update_counter += 1

    def get_loss(self, adam, memory, config, batch): 
        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.syn_sample(batch)
        states = torch.tensor(b_states).float().to(self.device)
        actions = torch.tensor(np.array(b_actions)).long().reshape(states.shape[0],-1,1).to(self.device)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1).to(self.device)
        current_q_values = self.q(states).gather(2, actions).squeeze(-1)
        expected_q_vals = torch.cat([rewards]*self.ac,dim=1)# + max_next_q_vals*self.gamma*masks

        loss = F.mse_loss(current_q_values, expected_q_vals)
        return loss


        


