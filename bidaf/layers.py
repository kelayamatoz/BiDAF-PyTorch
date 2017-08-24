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


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, \
                    filter_height, filter_width, is_train=None, \
                    keep_prob=1.0, padding=0):
        super(Conv1D, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.is_train = is_train
        self.keep_prob = keep_prob
        self.dropout_ = nn.Dropout(1. - keep_prob)
        self.padding = padding
        self.kernel_size = (filter_height, filter_width)
        self.filter_height = filter_height
        self.filter_width = filter_width
        # Tensorflow API:
        # input tensor of shape [batch, in_height, in_width, in_channels]
        # filter / kernel tensor of shape 
        # [filter_height, filter_width, in_channels, out_channels]
        # filter_height = 1, filter_width = height, num_channels = in_channels, out_channels = filter_size
        # Usage:
        # xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        # filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        # bias = tf.get_variable("bias", shape=[filter_size], dtype='float'

        # Pytorch API:
        # in_channels (int) – Number of channels in the input image
        # out_channels (int) – Number of channels produced by the convolution
        # kernel_size (int or tuple) – Size of the convolving kernel
        # stride (int or tuple, optional) – Stride of the convolution. Default: 1
        # padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        # dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        # groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        # bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        print((filter_height, filter_width))
        # self.conv2d_ = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, \
        #                             bias=True, padding=self.padding)



    def forward(self, in_):
        if self.is_train is not None and self.keep_prob < 1.0:
            self.dropout_(in_)
        # tf: input tensor of shape [batch, in_height, in_width, in_channels]
        # pt: input tensor of shape [batch, in_channels, in_height, in_width]
        t_in = in_.permute(0, 3, 1, 2)
        filter_ = Variable(torch.zeros(self.out_channels, self.in_channels, self.filter_height, self.filter_width))
        print("permuted_in_ size = " + str(t_in.size()))
        print('t_in shape = ', str(t_in.size()))
        # xxc = self.conv2d_(t_in)
        xxc = F.conv2d(t_in, filter_)
        # use desired inputs from pt to produce size information
        # Hout=floor((Hin+2∗padding[0]−dilation[0]∗(kernel_size[0]−1)−1)/stride[0]+1)
        # Wout=floor((Win+2∗padding[1]−dilation[1]∗(kernel_size[1]−1)−1)/stride[1]+1)
        n, c_in, h_in, w_in = t_in.size()
        kernel_size_0, kernel_size_1 = self.kernel_size
        d_Height = (h_in + 2 * 0. - 0. * (kernel_size_0 - 1) - 1) / 1 + 1
        d_Width = (w_in + 2 * 0. - 0. * (kernel_size_1 - 1) - 1) / 1 + 1
        print('xxc shape = ', str(xxc.size()), ', desired height = ', str(d_Height), ', desired width = ', str(d_Width))
        out, argmax_out = torch.max(F.relu(xxc), 2)
        return out


class MultiConv1D(nn.Module):
    def __init__(self, is_train, keep_prob):
        super(MultiConv1D, self).__init__()
        self.is_train = is_train
        self.keep_prob = keep_prob
        self.conv1d_list = nn.ModuleList()


    def forward(self, in_, filter_sizes, heights, padding):
        assert len(filter_sizes) == len(heights)
        if padding == 'VALID':
            padding_ = 0
        elif padding == 'SAME':
            padding_ = 0
            print('Warning: don\'t now how to set for \'SAME\' padding')
        else:
            raise Exception('Exception: unknown padding'+padding)

        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            print("filter_size = "+str(filter_size))
            print("height = "+str(height))
            if filter_size == 0:
                continue
            # in_ shape: batch, in_height, in_width, in_channels
            batch_size, in_height, in_width, in_channels = in_.size()
            filter_height = 1
            filter_width = height
            out_channels = filter_size
            self.conv1d_list.append(Conv1D(in_channels, out_channels, filter_height, filter_width, \
                                           is_train=self.is_train, keep_prob=self.keep_prob, padding=padding_))

        print('>>>>>>>>>> in_ shape = ', str(in_.size()))
        for conv1d_layer in self.conv1d_list:
            out = conv1d_layer(in_) 
            outs.append(out)

        concat_out = torch.cat(outs, 2)
        return concat_out


# TBA implemenations
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
