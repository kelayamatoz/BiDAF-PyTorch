import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import code

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def softsel(target, logits):
    out = F.softmax(logits)
    out = out.unsqueeze(len(out.size())).mul(target).sum(len(target.size())-2)
    return out


def exp_mask(logits, mask):
    return torch.add_(logits, (1 - mask)) * VERY_NEGATIVE_NUMBER


def softmax3d(input, xd, yd):
    out = input.view(-1, xd*yd)
    out = F.softmax(out).view(-1, xd, yd)
    return out


def span_loss(config, q_mask, logits_start, start, logits_end, end):
    size = config.max_num_sents * config.max_sent_size
    loss_mask = reduce_mask(q_mask, 1)
    losses_start = nn.CrossEntropyLoss(logits_start, start.view(-1, size))
    ce_loss_start = torch.mean(loss_mask * losses)
    losses_end = nn.CrossEntropyLoss(logits_end, end.view(-1, size))
    ce_loss_end = torch.mean(loss_mean)
    return ce_loss_end - ce_loss_start


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, \
                    filter_height, filter_width, is_train=None, \
                    keep_prob=1.0, padding=0):
        super(Conv1D, self).__init__()
        self.is_train = is_train
        self.dropout_ = nn.Dropout(1. - keep_prob)
        self.keep_prob = keep_prob
        kernel_size = (filter_height, filter_width)
        self.conv2d_ = nn.Conv2d(in_channels, out_channels, kernel_size, \
                                    bias=True, padding=padding)



    def forward(self, in_):
        if self.is_train is not None and self.keep_prob < 1.0:
            self.dropout_(in_)
        '''
        tf: input tensor of shape [batch, in_height, in_width, in_channels]
        pt: input tensor of shape [batch, in_channels, in_height, in_width]
        '''
        t_in = in_.permute(0, 3, 1, 2)
        xxc = self.conv2d_(t_in)
        out, argmax_out = torch.max(F.relu(xxc), -1)
        return out


class MultiConv1D(nn.Module):
    def __init__(self, is_train, keep_prob):
        super(MultiConv1D, self).__init__()
        self.is_train = is_train
        self.keep_prob = keep_prob
        self.conv1d_list = nn.ModuleList()


    def forward(self, in_, filter_sizes, heights, padding, is_shared=False):
        assert len(filter_sizes) == len(heights)
        if padding == 'VALID':
            padding_ = 0
        elif padding == 'SAME':
            padding_ = 0
            print('Warning: don\'t now how to set for \'SAME\' padding')
        else:
            raise Exception('Exception: unknown padding'+padding)

        outs = []
        for idx, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            print("filter_size = "+str(filter_size))
            print("height = "+str(height))
            if filter_size == 0:
                continue
            # in_ shape: batch, in_height, in_width, in_channels
            batch_size, in_height, in_width, in_channels = in_.size()
            filter_height = 1
            filter_width = height
            out_channels = filter_size
            '''
            Comment: Pytorch doesn't support reusable variables. However, we can reuse these
            variables by passing data through the same layers.
            '''
            if not is_shared:
                self.conv1d_list.append(Conv1D(in_channels, out_channels, filter_height, filter_width, \
                                                    is_train=self.is_train, keep_prob=self.keep_prob, padding=padding_))

        for conv1d_layer in self.conv1d_list:
            out = conv1d_layer(in_) 
            outs.append(out)

        concat_out = torch.cat(outs, 2)
        return concat_out


class HighwayLayer(nn.Module):
    # TODO: We may need to add weight decay here
    def __init__(self, size, bias_init=0.0, nonlin=nn.ReLU(inplace=True), gate_nonlin=F.sigmoid):
        super(HighwayLayer, self).__init__()

        self.nonlin = nonlin
        self.gate_nonlin = gate_nonlin
        self.lin = nn.Linear(size, size)
        self.gate_lin = nn.Linear(size, size)
        self.gate_lin.bias.data.fill_(bias_init)

    def forward(self, x):
        trans = self.nonlin(self.lin(x))
        gate = self.gate_nonlin(self.gate_lin(x))
        return torch.add(torch.mul(gate, trans), torch.mul((1 - gate), x))


class HighwayNet(nn.Module):
    def __init__(self, depth, size):
        super(HighwayNet, self).__init__()
        layers = [HighwayLayer(size) for _ in range(depth)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, do_p=0.2):
        super(LinearBase, self).__init__()
        self.do = nn.Dropout(do_p)
        self.lin = nn.Linear(input_size, output_size)
        self.input_size = input_size

    def forward(self, a, b, mask):
        shape = a.size()
        N = self.input_size
        M = a.numel() // size
        a_ = a.view(M, N)
        b_ = b.view(M, N)
        return shape, a_, b_


class Linear(LinearBase):
    def forward(self, a, b, mask):
        shape, a_, b_ = super(self).forward(a, b, mask)
        input = torch.cat((a_, b__), 1)
        out = self.lin(self.do(input))
        out = out.view(shape).squeeze(len(shape)-1)
        return exp_mask(out, mask)


class BiEncoder(nn.Module):
    def __init__(self, config, input_size):
        super(BiEncoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.hidden_size,
                           num_layers=config.lstm_layers, batch_first=config.batch_first, 
                           dropout=(1 - config.input_keep_prob),
                           bidirectional=True)
        print('input_size = ', str(input_size))
        print('hidden_size = ', str(config.hidden_size))

    def forward(self, inputs):
        batch_size, seq_len, feature_size = inputs.size()
        print('batch_size =  ', str(batch_size))
        print('seq_len =  ', str(seq_len))
        print('feature_size =  ', str(feature_size))

        # TODO: Would these two hidden variables requires grads?
        # What is a good initializer? 
        # h_0 = Variable(torch.zeros(batch_size, seq_len, feature_size), requires_grad=False)
        # c_0 = Variable(torch.zeros(batch_size, seq_len, feature_size), requires_grad=False)

        h_0 = Variable(torch.zeros(seq_len, batch_size, feature_size), requires_grad=False)
        c_0 = Variable(torch.zeros(seq_len, batch_size, feature_size), requires_grad=False)
        outputs, (h_n, c_n) = self.rnn(inputs, (h_0, c_0)) 
        return outputs


# TBA implemenations
class TriLinear(LinearBase):
    def forward(self, a, b, mask):
        shape, a_, b_ = super(self).forward(a, b, mask)
        input = torch.cat((a_, b_, a_*b_), 1)
        out = self.lin(self.do(input))
        out = out.view(shape).squeeze(len(shape)-1)
        return exp_mask(out, mask)


class TFLinear(nn.Module):
    def __init__(self, input_size, output_size, func, do_p=0.2):
        super(TFLinear, self).__init__()
        if func == 'linear':
            self.main = Linear(input_size, output_size, do_p)
        elif func == 'trilinear':
            self.main = TriLinear(input_size, output_size, do_p)
        else:
            assert False

    def forward(self, a, b, mask):
        return self.main(a, b, mask)
    
            
class FixedEmbedding(nn.Embedding):
    def forward(input):
        out = super(FixedEmbedding, self).forward(input)
        return Variable(out.data)


class BiAttention(nn.Module):
    def __init__(self, args, logits_size):
        super(BiAttention, self).__init__()
        self.lin = TFLinear(size, args.attn_func)
        self.args = args

    def forward(self, text, query, text_mask, query_mask):
        a = self.args
        max_sent_size, max_num_sents, max_q_size = \
            a.max_sent_size, a.max_num_sents, a.max_q_size
        text_aug = text.unsqueeze(3).repeat(1, 1, 1, max_q_size, 1)
        query_aug = query.unqueeze(1).unsqueeze(1).repeat(1, max_num_sents, max_sent_size, 1, 1)
        text_mask_aug = text_mask.unsqueeze(3).repeat(1, 1, 1, max_q_size)
        query_mask_aug = query_mask.unqueeze(1).unsqueeze(1).repeat(1, max_num_sents, max_sent_size, 1)
        text_query_mask = text_mask_aug * query_mask_aug
        query_logits = self.lin(text_aug, query_aug, text_query_mask)

        _, query_logits_max = torch.max(query_logits, 3)
        # c2q
        text_attn = softsel(text, query_logits_max).unsqueeze(2).repeat(1, 1, max_sent_size, 1)
        # q2c
        query_attn = softsel(query_aug, query_logits)

        attn = torch.cat((text, query_attn, text * query_attn, text * text_attn), 3)
        return attn
