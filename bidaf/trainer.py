import torch

class MultiGPUTrainer(object):
    def __init__(self, config, model=None):
        # assert isinstance(model, Model)
        self.config = config
        # self.model = model
        # self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        # self.loss = model.get_loss()
        # self.var_list = model.get_var_list()
        # self.global_step = model.get_global_step()
        # self.summary = model.summary
        # self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        # self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    # def get_train_op(self):
        # return self.train_op

    def step(self, batches, get_summary=False):
        # assert isinstance(sess, tf.Session)
        for batch in batches:
            _, ds = batch
            print(batch)
            print(ds)

        # feed_dict = self.model.get_feed_dict(ds, True)
        # if get_summary:
        #     loss, summary, train_op = \
        #         sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        # else:
        #     loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        #     summary = None
        # return loss, summary, train_op