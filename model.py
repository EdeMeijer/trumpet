import tensorflow as tf


def make_weight_variable(name, num_inputs, num_outputs):
    return tf.get_variable(
        name,
        [num_inputs, num_outputs],
        initializer=tf.contrib.layers.variance_scaling_initializer()
    )


class Model:
    def __init__(self, chars, max_steps, lstm_units=250, l1_units=200, l2_units=150,
                 learning_rate=0.001, l2=0.001):
        """

        """
        self.chars = chars
        self.max_steps = max_steps

        l1_weights = make_weight_variable("l1-weights", lstm_units, l1_units)
        l1_biases = tf.Variable(0.1, name='L1-biases')

        l2_weights = make_weight_variable("l2-weights", l1_units, l2_units)
        l2_biases = tf.Variable(0.1, name='L2-biases')

        out_weights = make_weight_variable("out-weights", l2_units, len(chars) + 1)
        out_biases = tf.Variable(0.1, name='out-biases')

        self.features = tf.placeholder(dtype=tf.int32, shape=[None, max_steps, 1])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, max_steps, 1])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None, max_steps])

        features_one_hot = tf.squeeze(
            tf.one_hot(self.features, len(chars) + 1, dtype=tf.float32, axis=2),
            axis=3
        )

        lstm_3d, _ = tf.nn.dynamic_rnn(
            cell=tf.contrib.rnn.LSTMCell(num_units=lstm_units, use_peepholes=True),
            dtype=tf.float32,
            inputs=features_one_hot
        )
        lstm_flat = tf.reshape(lstm_3d, [-1, lstm_units])

        layer1 = tf.nn.relu(tf.matmul(lstm_flat, l1_weights) + l1_biases)
        layer2 = tf.nn.relu(tf.matmul(layer1, l2_weights) + l2_biases)
        self.logits = tf.matmul(layer2, out_weights) + out_biases

        loss_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(tf.reshape(self.labels, [-1, 1]), axis=1),
            logits=self.logits
        )
        loss_flat_masked = loss_flat * tf.reshape(self.mask, [-1, 1])
        self.loss = tf.reduce_mean(loss_flat_masked)

        weight_vars = [v for v in tf.trainable_variables() if 'bias' not in v.name]
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_vars]) * l2

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss + self.l2_loss)
