import torch
import torch.optim as O
import os.path
import numpy as np
import random


# TODO: Need to pass in multiple models in the next iteration
class MultiGPUTrainer(object):
    def __init__(self, config, model):
        # assert isinstance(model, Model)
        self.config = config
        # TODO: self.models = models
        self.model = model
        # self.model = model
        # self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        # self.loss = model.get_loss()
        # self.var_list = model.get_var_list()
        # self.global_step = model.get_global_step()
        # self.summary = model.summary
        # self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        # self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    # def get_train_op(self)i
        # return self.train_op

    def step(self, batch, get_summary=False, supervised=True):
        # assert isinstance(sess, tf.Session)
        # TODO: for batch, model in zip(batches, self.models)
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')
        y = np.zeros([N, M, JX], dtype='bool')
        y2 = np.zeros([N, M, JX], dtype='bool')
    
        X = batch.data['x']
        CX = batch.data['cx']

        if supervised:
            for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    offset = sum(map(len, xi[:j]))
                    j, k = 0, k + offset
                    offset = sum(map(len, xi[:j2]))
                    j2, k2 = 0, k2 + offset
                y[i, j, k] = True
                y2[i, j2, k2-1] = True

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        inputs = [x, cx, x_mask, q, cq, q_mask]
        if config.use_glove_for_unk:
            new_emb_mat = batch.shared['new_emb_mat']
            inputs.append(new_emb_mat)

        print(self.model)
        self.model(*inputs)

        # feed_dict = self.model.get_feed_dict(ds, True)
        # if get_summary:
        #     loss, summary, train_op = \
        #         sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        # else:
        #     loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        #     summary = None
        # return loss, summary, train_op