def _build_loss(self):
    config = self.config
    JX = tf.shape(self.x)[2]
    M = tf.shape(self.x)[1]
    JQ = tf.shape(self.q)[1]
    loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        self.logits, tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
    ce_loss = tf.reduce_mean(loss_mask * losses)
    tf.add_to_collection('losses', ce_loss)
    ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        self.logits2, tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float')))
    tf.add_to_collection("losses", ce_loss2)

    self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
    tf.scalar_summary(self.loss.op.name, self.loss)
    tf.add_to_collection('ema/scalar', self.loss)