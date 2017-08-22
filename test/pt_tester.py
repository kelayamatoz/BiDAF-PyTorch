import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from torch.nn import Embedding
from torch.autograd import Variable
from torch import Tensor, from_numpy


dtype = torch.cuda.FloatTensor
ind_type = torch.cuda.LongTensor


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
        self.char_embed.weight = torch.nn.Parameter(Tensor(test_emb).type(dtype))
        self.conv2d_ = nn.Conv2d(8, 100, (1, 5))

    def forward(self, x):
        emb = self.char_embed(Variable(x))
        result = emb.view(-1,8,2,10)
        input = Variable(Tensor(6000, 50, 50, 8).type(dtype))
        t_in = input.permute(0, 3, 1, 2)
        xxc = self.conv2d_(result)
        result = torch.max(F.relu(xxc), 2)

        return result

model = Tester()
model.cuda(0)
# a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]).reshape(4,2)
a = np.zeros([4,2,2], dtype='int').reshape(8,2)
at = torch.LongTensor(a).cuda()
b = model(at)
print(b)
