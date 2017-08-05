import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


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


def reduce_max(input_tensor, axis):
    _, values = input_tensor.max(axis)
    return values


def span_loss(config, q_mask, logits_start, start, logits_end, end):
    size = config.max_num_sents * config.max_sent_size
    loss_mask = reduce_mask(q_mask, 1)
    losses_start = nn.CrossEntropyLoss(logits_start, start.view(-1, size))
    ce_loss_start = torch.mean(loss_mask * losses)
    losses_end = nn.CrossEntropyLoss(logits_end, end.view(-1, size))
    ce_loss_end = torch.mean(loss_mean)
    return ce_loss_end - ce_loss_start


class HighwayLayer(nn.Module):
    def __init__(self, size, bias_init=0.0, nonlin=nn.ReLU(inplace=True), gate_nonlin=F.sigmoid):
        super(HighwayLayer, self).__init__()

        self.nonlin = nonlin
        self.gate_nonlin = gate_nonlin
        self.lin = nn.Linear(size, size)
        self.gate_lin = nn.Linear(size, size)
        self.gate_lin.bias.data.fill_(bias_init)

    def forward(self, x):
        out = self.nonlin(self.lin(x))
        gate_out = self.gate_nonlin(self.gate_lin(x))
        prod = torch.mul(out, gate_out)
        resid = torch.mul((1-gate_out), x)
        return torch.add(prod, resid)


class HighwayNet(nn.Module):
    def __init__(self, size, depth):
        super(HighwayNet, self).__init__()
        layers = [HighwayLayer(size) for _ in range(depth)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Conv1dMax(nn.Module):
    def __init__(self, in_chan, out_chan, width, do_p=0.5):
        self.do = nn.Dropout(do_p)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=[1, width])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(self.do(x)))
        _, out = torch.max(out, 2)
        return out


class Conv1dN(nn.Module):
    def __init__(self, nchan, filter_sizes, filter_heights, do_p):
        super(Conv1dN, self).__init__()

        conv_layers = [Conv1dMax(nchan, size, height, do_p)
                       for size, height in zip(filter_size, filter_heights)]
        self.main = nn.Sequential(*conv_layers)

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
    
            
class BiEncoder(nn.Module):
    def __init__(self, config, input_size):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.dp_ratio,
                           bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, _ = self.rnn(inputs, (h0, c0))
        return outputs


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
