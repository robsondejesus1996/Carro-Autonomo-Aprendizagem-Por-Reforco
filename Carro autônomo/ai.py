# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:58:47 2022

@author: Robson de Jesus
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable



# estrutura da rede
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
            
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []