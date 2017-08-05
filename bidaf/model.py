import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable

# TODO: Multi GPU settings?

dtype = torch.FloatTensor

class BiDAF(nn.Module):
    def __init__(self, config):
        super(BiDAF, self).__init__()
        self.config = config
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        
    def forward(self, x, cx, x_mask, q, cq, q_mask, new_emb_mat=None):
        print("in forward function of BiDAF, printing x:")
        print(x)
        print("q")
        print(q)
        print("emb_mat")
        print(new_emb_mat)