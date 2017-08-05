import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable

# TODO: Multi GPU settings?

class BiDAF(nn.Module):
    def __init__(self, config):
        super(BiDAF, self).__init__()
        self.config = config

    def forward(self, x, cx, x_mask, q, cq, q_mask, new_emb_mat=None):
        print("in forward function of BiDAF, printing x:")
        print(x)
        print("q")
        print(q)