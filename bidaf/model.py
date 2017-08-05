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

        # params from TBA
        batch_size, d_hidden = config.batch_size, config.hidden_size
        max_num_sents, max_sent_size = config.max_num_sents, config.max_sent_size
        max_ques_size, max_word_size = config.max_ques_size, config.max_word_size
        word_vocab_size, char_vocab_size = config.word_vocab_size, config.char_vocab_size
        d_char_embed, d_embed = config.char_emb_size, config.glove_vec_size
        d_char_out = config.char_out_size


    def forward(self, x, cx, x_mask, q, cq, q_mask, new_emb_mat=None):
        print("in forward function of BiDAF, printing x:")
        print(x)
        print("q")
        print(q)
        print("emb_mat")
        print(new_emb_mat)