import tensorflow as tf
import numpy as np
import os


class CNN(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.learning_rate = 1e-3
        self.batch_size = 4
        self.n_epoch = 100000
        self.n_samples = 1000

        self.X = tf.placeholder("float", [None, 96, 1290, 1])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    def init_data(self):
        """
        Create convolution neural network for audio classification problem.
        :return: output layer, Saver class
        """
        weights = {
            'wconv1': self.init_weights([3, 3, 1, 32]),
            'wconv2': self.init_weights([3, 3, 32, 64]),
            'wconv3': self.init_weights([3, 3, 64, 128]),
            'wconv4': self.init_weights([3, 3, 128, 192]),
            'wconv5': self.init_weights([3, 3, 192, 256]),
            'bconv1': self.init_biases([32]),
            'bconv2': self.init_biases([64]),
            'bconv3': self.init_biases([128]),
            'bconv4': self.init_biases([192]),
            'bconv5': self.init_biases([256]),
            'woutput': self.init_weights([256, 10]),
            'boutput': self.init_biases([10])}
        x = self.batch_norm(tf.reshape(self.X, [-1, 1, 96, 1290]), 1290, self.phase_train)
        x = tf.reshape(x, [-1, 96, 1290, 1])
        conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
        conv2_1 = tf.nn.relu(self.batch_norm(conv2_1, 32, self.phase_train))
        mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
        dropout_1 = tf.nn.dropout(mpool_1, self.dropout_rate)  # 48, 333

        conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv2'])
        conv2_2 = tf.nn.relu(self.batch_norm(conv2_2, 64, self.phase_train))
        mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
        dropout_2 = tf.nn.dropout(mpool_2, self.dropout_rate)  # 24, 84

        conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv3'])
        conv2_3 = tf.nn.relu(self.batch_norm(conv2_3, 128, self.phase_train))
        mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
        dropout_3 = tf.nn.dropout(mpool_3, self.dropout_rate)  # 12, 21

        conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv4'])
        conv2_4 = tf.nn.relu(self.batch_norm(conv2_4, 192, self.phase_train))
        mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
        dropout_4 = tf.nn.dropout(mpool_4, self.dropout_rate)  # 4, 5

        conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'),
                         weights['bconv5'])
        conv2_5 = tf.nn.relu(self.batch_norm(conv2_5, 256, self.phase_train))
        mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
        dropout_5 = tf.nn.dropout(mpool_5, self.dropout_rate)  # 1, 1

        flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
        p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat, weights['woutput']), weights['boutput']))

        saver = tf.train.Saver(weights, max_to_keep=3)

        return p_y_X, saver

    @staticmethod
    def batch_norm(x, n_out, phase_train, scope='bn'):
        """
        TODO: complete doc
        :param x:
        :param n_out:
        :param phase_train:
        :param scope:
        :return:
        """
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

    def start_train(self):
        """
        TODO: complete doc
        :return:
        """
        y = tf.placeholder("float", [None, 10])
        lrate = tf.placeholder("float")
        out, saver = self.init_data()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                file_path, step = self.return_last_checkpoint()
                saver = tf.train.import_meta_graph('{}.meta'.format(file_path))
                saver.restore(sess, file_path)
                print(f"Found checkpoint under {file_path}. Continuing from this point...")
            except OSError as e:
                print("Any checkpoint was not found. Creating new checkpoint...")
            self.dataset.init_dataset()
            for i in range(step+1, self.n_epoch):
                print("Epoch {}".format(i))
                # train
                for _ in range(int(self.dataset["train"]["size"] / self.batch_size)):
                    batch = self.dataset.next_batch("train", self.batch_size)
                    train_input_dict = {self.X: batch[0],
                                        y: batch[1],
                                        lrate: self.learning_rate,
                                        self.phase_train: True,
                                        self.dropout_rate: 0.5}
                    sess.run(train_op, feed_dict=train_input_dict)
                if i % 5 == 0:
                    # validate every five epochs
                    saver.save(sess, './model/my-model.ckpt', global_step=i)
                    sum = 0
                    size = int(self.dataset["validate"]["size"] / self.batch_size)
                    for _ in range(size):
                        batch = self.dataset.next_batch("validate", self.batch_size)
                        test_input_dict = {self.X: batch[0], y: batch[1],
                                           self.phase_train: True, self.dropout_rate: 1.0}
                        sum += accuracy.eval(feed_dict=test_input_dict)
                    print("Validate:   {}".format(sum / size))

    def start_test(self):
        """
        TODO: complete doc
        :return:
        """
        y = tf.placeholder("float", [None, 10])
        out, _ = self.init_data()

        accuracy, accuracy_update = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(out, 1), name='accuracy')

        batch_confusion = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(out, 1), 10)
        confusion = tf.Variable(tf.zeros([10, 10], dtype=tf.int32), name='confusion')
        confusion_update = confusion.assign(confusion + batch_confusion)

        test_op = tf.group(accuracy_update, confusion_update)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            try:
                file_path, _ = self.return_last_checkpoint()
                saver = tf.train.import_meta_graph(f'./model/{file_path}.meta')
                saver.restore(sess, f"./model/{file_path}")
                print(f"Found checkpoint under {file_path}. Starting test...")
            except OSError:
                print("Any checkpoint was not found. Exiting now...")
                return
            self.dataset.init_test()
            size = int(self.dataset["test"]["size"] / self.batch_size)
            for _ in range(size):
                batch = self.dataset.next_batch("test", self.batch_size)
                test_input_dict = {self.X: batch[0], y: batch[1], self.phase_train: True, self.dropout_rate: 1.0}
                sess.run(test_op, feed_dict=test_input_dict)
            result = sess.run(accuracy)
            print(f"Test: {result}%")
            conf_array = confusion.eval(sess)
            np.savetxt("model/matrix.txt", conf_array)
            print("Confusion matrix saved under ./model/matrix.txt file")

    @staticmethod
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    @staticmethod
    def init_biases(shape):
        return tf.Variable(tf.zeros(shape))

    @staticmethod
    def return_last_checkpoint():
        """
        TODO: complete doc
        :return:
        """
        checkpoint_dir = "./model/"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[2])
        return ckpt.model_checkpoint_path, step
