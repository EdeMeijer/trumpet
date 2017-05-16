import json
import math
import os
import random

import numpy as np
import tensorflow as tf

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'
BATCH_SIZE = 128
L1_UNITS = 512
L2_UNITS = 256

TEMPERATURE = 0.8
L2_WEIGHT = 0.00005


def split_test_train(data):
    test_examples = round(data.shape[0] * 0.2)
    return data[:test_examples], data[test_examples:]


features_test, features_train = split_test_train(np.load(CACHE_DIR + '/features.npy'))
labels_test, labels_train = split_test_train(np.load(CACHE_DIR + '/labels.npy'))
mask_test, mask_train = split_test_train(np.load(CACHE_DIR + '/mask.npy'))

with open(CACHE_DIR + '/settings.json') as file:
    settings = json.load(file)

max_steps = settings['maxSteps']
chars = settings['chars']

NUM_OUTPUTS = len(chars) + 1

# =========== GRAPH ===========
weights = {
    'L1': tf.Variable(tf.truncated_normal([L1_UNITS, L2_UNITS])),
    'L2': tf.Variable(tf.truncated_normal([L2_UNITS, NUM_OUTPUTS])),
}
biases = {
    'L1': tf.Variable(0.1),
    'L2': tf.Variable(0.1),
}

x = tf.placeholder(dtype=tf.int32, shape=[None, max_steps, 1], name='x')
y = tf.placeholder(dtype=tf.int32, shape=[None, max_steps, 1], name='y')
mask = tf.placeholder(dtype=tf.float32, shape=[None, max_steps], name='mask')

char_feature_one_hot = tf.squeeze(tf.one_hot(x, len(chars), dtype=tf.float32, axis=2), axis=3)

char_labels_flat = tf.reshape(y, [-1, 1])

with tf.variable_scope("L1"):
    L1_output_full, _ = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(num_units=L1_UNITS, use_peepholes=True),
        dtype=tf.float32,
        inputs=char_feature_one_hot
    )
    L1_output_flat = tf.reshape(L1_output_full, [-1, L1_UNITS])
    L1_activation_flat = tf.nn.elu(tf.matmul(L1_output_flat, weights['L1']) + biases['L1'])

with tf.variable_scope("L2"):
    L2_activation_flat = tf.nn.elu(tf.matmul(L1_activation_flat, weights['L2']) + biases['L2'])

predictions_flat = tf.nn.softmax(L2_activation_flat / TEMPERATURE, 1)

loss_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.squeeze(char_labels_flat, axis=1),
    logits=L2_activation_flat
)
loss_flat_masked = loss_flat * tf.reshape(mask, [-1, 1])
loss = tf.reduce_mean(loss_flat_masked)

l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * L2_WEIGHT

train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss + l2_loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


def sample_tweet():
    # Start with the start symbol, which has label num_chars
    input = [len(chars)]
    tweet = ''

    for i in range(300):
        next = sample_next_char(input[-143:])
        input.append(next)
        if next == len(chars):
            break
        tweet += chars[next]

    return tweet.strip()


def sample_next_char(classes):
    sample_input = np.zeros([1, max_steps, 1])
    sample_input[:, :len(classes)] = np.array(classes).reshape([1, len(classes), 1])

    predictions = sess.run(
        predictions_flat,
        feed_dict={
            x: sample_input,
        }
    )
    probabilities = predictions[len(classes) - 1]
    rnd = random.random()
    accum = 0

    for idx in range(len(probabilities)):
        accum += probabilities[idx]
        if accum >= rnd:
            return idx
    print('PROBS DONT SUM TO 1.0!', np.sum(probabilities))
    return np.argmax(classes)


def calc_test_error():
    sum = 0
    weight = 0
    num_examples = features_test.shape[0]
    num_batches = math.ceil(num_examples / BATCH_SIZE)
    for batch in range(num_batches):
        start = batch * BATCH_SIZE
        end = (batch + 1) * BATCH_SIZE
        batch_x = features_test[start:end]
        err = sess.run(
            loss,
            feed_dict={
                x: batch_x,
                y: labels_test[start:end],
                mask: mask_test[start:end]
            }
        )
        batch_examples = batch_x.shape[0]
        weight += batch_examples
        sum += err * batch_examples
    return sum / weight


def train_epoch(epoch):
    num_examples = features_train.shape[0]
    num_batches = math.ceil(num_examples / BATCH_SIZE)
    for batch in range(num_batches):
        start = batch * BATCH_SIZE
        end = (batch + 1) * BATCH_SIZE
        _, err, l2 = sess.run(
            [train_op, loss, l2_loss],
            feed_dict={
                x: features_train[start:end],
                y: labels_train[start:end],
                mask: mask_train[start:end]
            }
        )
        print(err, l2)
    print('EPOCH {} -> {}'.format(epoch, calc_test_error()))
    print('Sampling tweet....')
    print('')
    print('---------------------------------------')
    print(sample_tweet())
    print('---------------------------------------')
    print('')


saver = tf.train.Saver()

print('EPOCH -1 -> ', calc_test_error())
for e in range(0, 300):
    train_epoch(e)
    save_path = saver.save(sess, CACHE_DIR + "/model.ckpt", global_step=e)
    print("Model saved in file: %s" % save_path)
