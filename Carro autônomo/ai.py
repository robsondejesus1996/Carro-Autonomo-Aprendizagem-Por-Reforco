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