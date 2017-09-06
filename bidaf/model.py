import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import bidaf.layers as L
from bidaf.layers import softsel
import numpy as np
import logging
import code


from numpy import genfromtxt
from torch.autograd import Variable
from torch.nn import Embedding
from torch import zeros, from_numpy, Tensor, LongTensor, FloatTensor
from argparse import ArgumentParser


dtype = torch.FloatTensor
ldtype = torch.LongTensor


PADDING = 'VALID'


class BiDAF(nn.Module):
    def __init__(self, config):
        super(BiDAF, self).__init__()
        self.config = config
        self.logits = None
        self.yp = None
        self.dc, self.dw, self.dco = config.char_emb_size, config.word_emb_size, \
                                        config.char_out_size
        self.N, self.M, self.JX, self.JQ, self.VW, self.VC, self.d, self.W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, \
            config.hidden_size, config.max_word_size
        self.word_embed = Embedding(config.word_vocab_size, \
                                           config.glove_vec_size)
        self.char_embed = Embedding(config.char_vocab_size, \
                                           config.char_emb_size)

        # char-level convs
        filter_sizes = list(map(int, config.out_channel_dims.split(',')))
        heights = list(map(int, config.filter_heights.split(',')))
        self.filter_sizes = filter_sizes
        self.heights = heights
        self.multiconv_1d = L.MultiConv1D(config.is_train, config.keep_prob)
        self.multiconv_1d_qq = L.MultiConv1D(config.is_train, config.keep_prob)
        if config.use_char_emb:
            highway_outsize = self.dco + self.dw
        else:
            highway_outsize = self.dw
        self.highway = L.HighwayNet(config.highway_num_layers, highway_outsize)
        self.prepro = L.BiEncoder(config, highway_outsize, hidden_size=config.hidden_size)
        self.prepro_x = L.BiEncoder(config, highway_outsize, hidden_size=config.hidden_size)
        self.attention_layer = L.AttentionLayer(config, self.JX, self.M, self.JQ, 2 * config.hidden_size)
        # Because p0 = torch.cat([h, u_a, torch.mul(h, u_a), torch.mul(h, h_a)], 3) and the last dim of 
        # these matrices are d.
        self.g0_biencoder = L.BiEncoder(config, 8 * config.hidden_size, hidden_size=config.hidden_size)
        self.g1_biencoder = L.BiEncoder(config, 2 * config.hidden_size, hidden_size=config.hidden_size)
        # p0: 8 * d. g1: 2 * d
        self.g1_logits = L.GetLogits(config, 10 * config.hidden_size, input_keep_prob=config.input_keep_prob, function=config.answer_func)
        # p0: [60, 1, 161, 800], g1: [60, 1, 161, 200], a1: [60, 1, 161, 200], g1 * a1: [60, 1, 161, 200]
        self.g2_biencoder = L.BiEncoder(config, 14 * config.hidden_size, hidden_size=config.hidden_size)
        # p0: 8 * d. g2: 2 * d
        self.g2_logits = L.GetLogits(config, 10 * config.hidden_size, input_keep_prob=config.input_keep_prob, function=config.answer_func)

    def forward(self, x, cx, x_mask, q, cq, q_mask, new_emb_mat):
        config = self.config
        filter_sizes = self.filter_sizes
        heights = self.heights
        dc = self.dc
        dw = self.dw
        dco = self.dco
        N = self.N
        VW = self.VW
        VC = self.VC
        d = self.d
        W = self.W

        JX = x.shape[2]
        JQ = q.shape[1]
        M = x.shape[1]
        N = x.shape[0]

        def get_long_tensor(np_tensor):
            if torch.cuda.is_available():
                return LongTensor(from_numpy(np_tensor)).cuda()
            else:
                return LongTensor(from_numpy(np_tensor))

        self.x = get_long_tensor(x.reshape(N, -1)) 
        self.cx = get_long_tensor(cx.reshape(N, -1))
        self.x_mask = get_long_tensor(x_mask)
        self.q_mask = get_long_tensor(q_mask)
        self.q = get_long_tensor(q.reshape(N, -1))
        self.cq = get_long_tensor(cq.reshape(N, -1))
        self.new_emb_mat = Tensor(new_emb_mat).type(dtype) 
        
        # Char Embedding Layer
        # TODO: Send this part to the layers.py
        if config.use_char_emb:
            '''
                Warning: currently, embedding only looks up 2-D inputs. To make 
                the embedding work as expected, I needed to flatten it first and 
                reshape it after going through the embedding layer.
            '''

            Acx_ = self.char_embed(Variable(self.cx)).view(N, M, JX, W, dc)
            Acx = Acx_.view(-1, JX, W, dc)
            Acq_ = self.char_embed(Variable(self.cq)).view(N, JQ, W, dc)
            Acq = Acq_.view(-1, JQ, W, dc)
            assert sum(filter_sizes) == dco, (filter_sizes, dco)
            xx_ = self.multiconv_1d(Acx, filter_sizes, heights, PADDING)
            if config.share_cnn_weights:
                qq_ = self.multiconv_1d(Acq, filter_sizes, heights, PADDING, is_shared=True)
            else: 
                qq_ = self.multiconv_1d_qq(Acq, filter_sizes, heights, PADDING)
            xx = xx_.view(-1, M, JX, dco)
            qq = qq_.view(-1, JQ, dco)

        if config.use_word_emb:
            if config.mode == 'train':
                word_emb_mat = Variable(Tensor(config.emb_mat))
            else:
                word_emb_mat = Variable(Tensor(Vw, dw).type(dtype))
            if config.use_glove_for_unk:
                word_emb_mat = torch.cat((word_emb_mat, self.new_emb_mat), 1)

            Ax = self.word_embed(Variable(self.x)).view(N, M, JX, dw)
            Aq = self.word_embed(Variable(self.q)).view(N, JQ, dw)

        if config.use_char_emb:
            xx = torch.cat((xx, Ax), 3)
            qq = torch.cat((qq, Aq), 2)
        else:
            xx = Ax
            qq = Aq

        # Highway network
        if config.highway:
            '''
            Warning: From the original tf implementation, it seems that xx and qq 
            go through the same highway network
            '''
            xx = self.highway(xx)
            qq = self.highway(qq)

        '''
        Warning: In tensorflow, the weights in LSTM are combined together into 
        a bigger matrix and then pass through the RNN. 
        TODO: This could possibly be an optimization in spatial LSTM implementation 
        size analysis:
        xx: [batch_size, max_num_sents, max_sent_size, di]
        qq: [batch_size, max_ques_size, di]
        '''
        xx = xx.view(N, M * JX, -1) # [batch_size, sequence, feature]
        xx = xx.permute(1, 0, 2)
        qq = qq.permute(1, 0, 2)
        u = self.prepro(qq)
        if config.share_lstm_weights:
            h = self.prepro(xx)
        else:
            h = self.prepro_x(xx)

        h = h.permute(1, 0, 2).contiguous().view(N, M, JX, -1) # [N, M, JX, 2 * d]
        u = u.permute(1, 0, 2) # [N, JQ, 2 * d]

        p0 = self.attention_layer(h, u, h_mask=self.x_mask, u_mask=self.q_mask) # (N, M, JX, 8 * d)
        g0 = self.g0_biencoder(p0.view(N, M * JX, -1)) # (N, M, JX, 2d)
        g1 = self.g1_biencoder(g0.view(N, M * JX, -1))
        logits = self.g1_logits((g1, p0), self.x_mask.squeeze())

        a1i = softsel(g1.view(N, M * JX, 2 * d), logits.view(N, M * JX))
        a1i = a1i.unsqueeze(1).unsqueeze(1).repeat(1, M, JX, 1)

        g1 = g1.view(N, M, JX, -1)
        g2 = self.g2_biencoder(torch.cat([p0, g1, a1i, g1 * a1i], 3).squeeze())
        logits2 = self.g2_logits((g2, p0), self.x_mask.squeeze())

        flat_logits = logits.view(-1, M * JX) 
        flat_yp = F.softmax(flat_logits)
        flat_logits2 = logits2.view(-1, M * JX)
        flat_yp2 = F.softmax(flat_logits2)

        if config.na:
            print("na case not implemented!")
            na_bias_tiled = Variable(Tensor(1,1).type(dtype)).repeat(N, 1)
            concat_flat_logits = torch.cat([na_bias_tiled, flat_logits], 1)
            concat_flat_yp = F.softmax(concat_flat_logits)
            print(concat_flat_yp.size())

        yp = flat_yp.view(-1, M, JX)
        yp2 = flat_yp2.view(-1, M, JX)
        wyp = F.sigmoid(logits2)

        return yp, yp2


if __name__ == '__main__':
    print("testing correctness of the model")
    flags = ArgumentParser(description='Model Tester')
    flags.add_argument("--max_num_sents", type=int, default=1)
    flags.add_argument("--max_sent_size", type=int, default=740)
    flags.add_argument("--max_ques_size", type=int, default=36)
    flags.add_argument("--word_vocab_size", type=int, default=1229)
    flags.add_argument("--char_vocab_size", type=int, default=281)
    flags.add_argument("--max_word_size", type=int, default=16)
    flags.add_argument("--glove_vec_size", type=int, default=100)
    flags.add_argument("--word_emb_size", type=int, default=100)

    flags.add_argument("--model_name", type=str, default="basic", help="Model name [basic]")
    flags.add_argument("--data_dir", type=str, default="data/squad", help="Data dir [data/squad]")
    flags.add_argument("--run_id", type=str, default="0", help="Run ID [0]")
    flags.add_argument("--out_base_dir", type=str, default="out", help="out base dir [out]")
    flags.add_argument("--forward_name", type=str, default="single", help="Forward name [single]")
    flags.add_argument("--answer_path", type=str, default="", help="Answer path []")
    flags.add_argument("--eval_path", type=str, default="", help="Eval path []")
    flags.add_argument("--load_path", type=str, default="", help="Load path []")
    flags.add_argument("--shared_path", type=str, default="", help="Shared path []")

    # Device placement flags.add_argument("--device", type=str, default="/cpu:0", help="default device for summing gradients. [/cpu:0]")
    flags.add_argument("--device_type", type=str, default="gpu", help="device for computing gradients (parallelization). cpu | gpu [gpu]")
    flags.add_argument("--num_gpus", type=int, default=1, help="num of gpus or cpus for computing gradients [1]")

    # Essential training and test options
    flags.add_argument("--mode", type=str, default="train", help="train | test | forward [test]")
    flags.add_argument("--load", type=bool, default=True, help="load saved data? [True]")
    flags.add_argument("--single", type=bool, default=False, help="supervise only the answer sentence? [False]")
    flags.add_argument("--debug", default=False, action="store_true", help="Debugging mode? [False]")
    flags.add_argument("--load_ema", type=bool, default=True, help="load exponential average of variables when testing?  [True]")
    flags.add_argument("--eval", type=bool, default=True, help="eval? [True]")
    flags.add_argument("--wy", type=bool, default=False, help="Use wy for loss / eval? [False]")
    flags.add_argument("--na", type=bool, default=False, help="Enable no answer strategy and learn bias? [False]")
    flags.add_argument("--th", type=float, default=0.5, help="Threshold [0.5]")

    # Training / test parameters
    flags.add_argument("--batch_size", type=int, default=60, help="Batch size [60]")
    flags.add_argument("--val_num_batches", type=int, default=100, help="validation num batches [100]")
    flags.add_argument("--test_num_batches", type=int, default=0, help="test num batches [0]")
    flags.add_argument("--num_epochs", type=int, default=12, help="Total number of epochs for training [12]")
    flags.add_argument("--num_steps", type=int, default=20000, help="Number of steps [20000]")
    flags.add_argument("--load_step", type=int, default=0, help="load step [0]")
    flags.add_argument("--init_lr", type=float, default=0.5, help="Initial learning rate [0.5]")
    flags.add_argument("--input_keep_prob", type=float, default=0.8, help="Input keep prob for the dropout of LSTM weights [0.8]")
    flags.add_argument("--keep_prob", type=float, default=0.8, help="Keep prob for the dropout of Char-CNN weights [0.8]")
    flags.add_argument("--wd", type=float, default=0.0, help="L2 weight decay for regularization [0.0]")
    flags.add_argument("--hidden_size", type=int, default=100, help="Hidden size [100]")
    flags.add_argument("--char_out_size", type=int, default=100, help="char-level word embedding size [100]")
    flags.add_argument("--char_emb_size", type=int, default=8, help="Char emb size [8]")
    flags.add_argument("--out_channel_dims", type=str, default="100", help="Out channel dims of Char-CNN, separated by commas [100]")
    flags.add_argument("--filter_heights", type=str, default="5", help="Filter heights of Char-CNN, separated by commas [5]")
    flags.add_argument("--finetune", type=bool, default=False, help="Finetune word embeddings? [False]")
    flags.add_argument("--highway", type=bool, default=True, help="Use highway? [True]")
    flags.add_argument("--highway_num_layers", type=int, default=2, help="highway num layers [2]")
    flags.add_argument("--share_cnn_weights", type=bool, default=True, help="Share Char-CNN weights [True]")
    flags.add_argument("--share_lstm_weights", type=bool, default=True, help="Share pre-processing (phrase-level) LSTM weights [True]")
    flags.add_argument("--var_decay", type=float, default=0.999, help="Exponential moving average decay for variables [0.999]")
    flags.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers")
    flags.add_argument("--batch_first", type=bool, default=True, help="LSTM order: (batch, seq, feature)")


    # Optimizations
    flags.add_argument("--cluster", type=bool, default=False, help="Cluster data for faster training [False]")
    flags.add_argument("--len_opt", type=bool, default=False, help="Length optimization? [False]")
    flags.add_argument("--cpu_opt", type=bool, default=False, help="CPU optimization? GPU computation can be slower [False]")

    # Logging and saving options
    flags.add_argument("--progress", type=bool, default=True, help="Show progress? [True]")
    flags.add_argument("--log_period", type=int, default=100, help="Log period [100]")
    flags.add_argument("--eval_period", type=int, default=1000, help="Eval period [1000]")
    flags.add_argument("--save_period", type=int, default=1000, help="Save Period [1000]")
    flags.add_argument("--max_to_keep", type=int, default=20, help="Max recent saves to keep [20]")
    flags.add_argument("--dump_eval", type=bool, default=True, help="dump eval? [True]")
    flags.add_argument("--dump_answer", type=bool, default=True, help="dump answer? [True]")
    flags.add_argument("--vis", type=bool, default=False, help="output visualization numbers? [False]")
    flags.add_argument("--dump_pickle", type=bool, default=True, help="Dump pickle instead of json? [True]")
    flags.add_argument("--decay", type=float, default=0.9, help="Exponential moving average decay for lobgging values [0.9]")

    # Thresholds for speed and less memory usage
    flags.add_argument("--word_count_th", type=int, default=10, help="word count th [100]")
    flags.add_argument("--char_count_th", type=int, default=50, help="char count th [500]")
    flags.add_argument("--sent_size_th", type=int, default=400, help="sent size th [64]")
    flags.add_argument("--num_sents_th", type=int, default=8, help="num sents th [8]")
    flags.add_argument("--ques_size_th", type=int, default=30, help="ques size th [32]")
    flags.add_argument("--word_size_th", type=int, default=16, help="word size th [16]")
    flags.add_argument("--para_size_th", type=int, default=256, help="para size th [256]")

    # Advanced training options
    flags.add_argument("--lower_word", type=bool, default=True, help="lower word [True]")
    flags.add_argument("--squash", type=bool, default=False, help="squash the sentences into one? [False]")
    flags.add_argument("--swap_memory", type=bool, default=True, help="swap memory? [True]")
    flags.add_argument("--data_filter", type=str, default="max", help="max | valid | semi [max]")
    flags.add_argument("--use_glove_for_unk", type=bool, default=True, help="use glove for unk [False]")
    flags.add_argument("--known_if_glove", type=bool, default=True, help="consider as known if present in glove [False]")
    flags.add_argument("--logit_func", type=str, default="tri_linear", help="logit func [tri_linear]")
    flags.add_argument("--answer_func", type=str, default="linear", help="answer logit func [linear]")
    flags.add_argument("--sh_logit_func", type=str, default="tri_linear", help="sh logit func [tri_linear]")

    # Ablation options
    flags.add_argument("--use_char_emb", type=bool, default=True, help="use char emb? [True]")
    flags.add_argument("--use_word_emb", type=bool, default=True, help="use word embedding? [True]")
    flags.add_argument("--q2c_att", type=bool, default=True, help="question-to-context attention? [True]")
    flags.add_argument("--c2q_att", type=bool, default=True, help="context-to-question attention? [True]")
    flags.add_argument("--dynamic_att", type=bool, default=False, help="Dynamic attention [False]")

    config = flags.parse_args()

    # Test with meta data from the first batch
    config.max_sent_size = 161
    config.max_ques_size = 20

    N, M, JX, JQ, VW, VC, d, W = \
    config.batch_size, config.max_num_sents, config.max_sent_size, \
    config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size

    config.emb_mat = genfromtxt('emb_mat.csv', delimiter=',')

    print(" >>>>>>>>>> DIMENSIONS <<<<<<<<<< ")
    print('N = ' + str(N))
    print('M = ' + str(M))
    print('JX = ' + str(JX))
    print('JQ = ' + str(JQ))
    print('VW = ' + str(VW))
    print('VC = ' + str(VC))
    print('d = ' + str(d))
    print('W = ' + str(W))
    print(" >>>>>>>>>> DIMENSIONS <<<<<<<<<< ")

    x = np.zeros([N, M, JX], dtype='int')
    cx = np.zeros([N, M, JX, W], dtype='int')
    x_mask = np.ones([N, M, JX], dtype='int')
    q = np.zeros([N, JQ], dtype='int')
    cq = np.zeros([N, JQ, W], dtype='int')
    q_mask = np.ones([N, JQ], dtype='int')
    y = np.zeros([N, M, JX], dtype='int')
    y2 = np.zeros([N, M, JX], dtype='int')
    new_emb_mat = np.zeros([VW, d], dtype='float')

    config.is_train = True
    model = BiDAF(config)

    if torch.cuda.is_available():
        print("cuda is available")
        model.cuda()

    inputs = [x, cx, x_mask, q, cq, q_mask, new_emb_mat]
    start, end = model(*inputs)
    print('start = ', start)
    print('end = ', end)
