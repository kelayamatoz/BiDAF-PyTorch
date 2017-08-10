import torch
import torch.nn as nn
import numpy as np

from torch.nn import Embedding
from torch.autograd import Variable
from torch import Tensor, from_numpy

class Tester(nn.Module):
    def __init__(self):
        super(Tester, self).__init__()
        word_vocab_size = 100
        glove_vec_size = 10
        ctr = 0 

        test_emb = np.ones([word_vocab_size, glove_vec_size])
        for i in range(word_vocab_size):
            for j in range(glove_vec_size):
                test_emb[i,j] = ctr
            ctr += 1

        self.char_embed = Embedding(word_vocab_size, glove_vec_size)
        self.char_embed.weight = torch.nn.Parameter(Tensor(test_emb))

    def forward(self, x):
        emb = self.char_embed(Variable(x))
        result = emb.view(-1,2,2,10)
        return result

model = Tester()
# a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]).reshape(4,2)
a = np.zeros([2,2,2], dtype='int').reshape(4,2)
at = torch.LongTensor(a)
b = model(at)
print(b)
