import tensorflow as tf


class CRNN(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.learning_rate = 1e-3
        self.batch_size = 4
        self.n_epoch = 1000
        self.n_samples = 1000

    def init_data(self):
        self.X = tf.placeholder("float", [None, 96, 2584, 1])  # 1366
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        weights = {
            'wconv1': self.init_weights([3, 3, 1, 32]),
            'wconv2': self.init_weights([3, 3, 32, 128]),
            'wconv3': self.init_weights([3, 3, 128, 128]),
            'wconv4': self.init_weights([3, 3, 128, 192]),
            'wconv5': self.init_weights([3, 3, 192, 256]),
            'wconv6': self.init_weights([3, 3, 256, 256]),
            'bconv1': self.init_biases([32]),
            'bconv2': self.init_biases([128]),
            'bconv3': self.init_biases([128]),
            'bconv4': self.init_biases([192]),
            'bconv5': self.init_biases([256]),
            'bconv6': self.init_biases([256]),
            'woutput': self.init_weights([256, 10]),
            'boutput': self.init_biases([10])}
        x = self.batch_norm(tf.reshape(self.X, [-1, 1, 96, 2584]), 2584, self.phase_train) # 2592
        x = tf.reshape(x, [-1, 96, 2584, 1])
        conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
        conv2_1 = tf.nn.relu(self.batch_norm(conv2_1, 32, self.phase_train))
        mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
        dropout_1 = tf.nn.dropout(mpool_1, 0.5)

        conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv2'])
        conv2_2 = tf.nn.relu(self.batch_norm(conv2_2, 128, self.phase_train))
        mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
        dropout_2 = tf.nn.dropout(mpool_2, 0.5)

        conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv3'])
        conv2_3 = tf.nn.relu(self.batch_norm(conv2_3, 128, self.phase_train))
        mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
        dropout_3 = tf.nn.dropout(mpool_3, 0.5)

        conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv4'])
        conv2_4 = tf.nn.relu(self.batch_norm(conv2_4, 192, self.phase_train))
        mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
        dropout_4 = tf.nn.dropout(mpool_4, 0.5)

        conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv5'])
        conv2_5 = tf.nn.relu(self.batch_norm(conv2_5, 256, self.phase_train))
        mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
        dropout_5 = tf.nn.dropout(mpool_5, 0.5)

        conv2_6 = tf.add(tf.nn.conv2d(dropout_5, weights['wconv6'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv6'])
        conv2_6 = tf.nn.relu(self.batch_norm(conv2_6, 256, self.phase_train))
        mpool_6 = tf.nn.max_pool(conv2_6, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
        dropout_6 = tf.nn.dropout(mpool_6, 0.5)

        flat = tf.reshape(dropout_6, [-1, weights['woutput'].get_shape().as_list()[0]])
        p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat, weights['woutput']), weights['boutput']))

        saver = tf.train.Saver(weights, max_to_keep=3)

        # gru1_in = tf.reshape(dropout_4, [-1, 27, 128])  # 27 warstw po 128 features -> 32 komórki (64 wagi, 32 ukryte, 32 wejscia)
        # gru1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 27)
        # gru1_out, state = tf.nn.dynamic_rnn(gru1, gru1_in, dtype=tf.float32, scope='gru1')
        #
        # gru2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 27)
        # gru2_out, state = tf.nn.dynamic_rnn(gru2, gru1_out, dtype=tf.float32, scope='gru2')
        # gru2_out = tf.transpose(gru2_out, [1, 0, 2])
        # gru2_out = tf.gather(gru2_out, int(gru2_out.get_shape()[0]) - 1)
        # dropout_5 = tf.nn.dropout(gru2_out, 0.3)
        #
        # flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
        # p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat, weights['woutput']), weights['boutput']))
        return p_y_X, saver

    def batch_norm(self, x, n_out, phase_train, scope='bn'):
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def start(self):
        y = tf.placeholder("float", [None, 10])
        lrate = tf.placeholder("float")
        out, saver = self.init_data()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.n_epoch):
                for _ in range(int(self.dataset["train"]["size"] / self.batch_size)):
                    batch = self.dataset.next_batch("train", self.batch_size)
                    train_input_dict = {self.X: batch[0],
                                        y: batch[1],
                                        lrate: self.learning_rate,
                                        self.phase_train: True}
                    sess.run(train_op, feed_dict=train_input_dict)
                if i > 0 and i % 5 == 0:
                    saver.save(sess, 'my-model', global_step=i)
                    sum = 0
                    size = int(self.dataset["validate"]["size"] / self.batch_size)
                    for _ in range(size):
                        batch = self.dataset.next_batch("validate", self.batch_size)
                        test_input_dict = {self.X: batch[0], y: batch[1], self.phase_train: True}
                        sum += accuracy.eval(feed_dict=test_input_dict)
                    print("Validate: {}".format(sum / size))

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def init_biases(self, shape):
        return tf.Variable(tf.zeros(shape))


if __name__ == "__main__":
    net = CRNN({})
    _ = net.init_data()