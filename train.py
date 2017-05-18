import json
import math
import os

import numpy as np
import tensorflow as tf

from model import Model
from tweet_sampler import TweetSampler

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'
BATCH_SIZE = 64

TEMPERATURE = 0.8
NUM_EPOCHS = 200


def split_test_train(data):
    test_examples = round(data.shape[0] * 0.1)
    return data[:test_examples], data[test_examples:]


features_test, features_train = split_test_train(np.load(CACHE_DIR + '/features.npy'))
labels_test, labels_train = split_test_train(np.load(CACHE_DIR + '/labels.npy'))
mask_test, mask_train = split_test_train(np.load(CACHE_DIR + '/mask.npy'))

with open(CACHE_DIR + '/settings.json') as file:
    settings = json.load(file)

model = Model(
    settings['chars'],
    settings['maxSteps'],
    l2=0.00005
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sampler = TweetSampler(sess, model, temperature=0.8)


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
            model.loss,
            feed_dict={
                model.features: batch_x,
                model.labels: labels_test[start:end],
                model.mask: mask_test[start:end]
            }
        )
        batch_examples = batch_x.shape[0]
        weight += batch_examples
        sum += err * batch_examples
    return sum / weight


def output_tweet_sample():
    print('Sampling tweet....')
    print('')
    print('---------------------------------------')
    print(sampler.sample())
    print('---------------------------------------')
    print('')


def train_epoch(epoch):
    num_examples = features_train.shape[0]
    num_batches = math.ceil(num_examples / BATCH_SIZE)
    total_err = 0
    total_l2 = 0
    for batch in range(num_batches):
        start = batch * BATCH_SIZE
        end = (batch + 1) * BATCH_SIZE
        _, err, l2 = sess.run(
            [model.train_op, model.loss, model.l2_loss],
            feed_dict={
                model.features: features_train[start:end],
                model.labels: labels_train[start:end],
                model.mask: mask_train[start:end]
            }
        )
        total_err += err
        total_l2 += l2
    print(
        'EPOCH {}: train = {}, test = {}, L2 = {}'.format(epoch, total_err / num_batches,
                                                          calc_test_error(),
                                                          total_l2 / num_batches))
    output_tweet_sample()


saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)

print('EPOCH -1 -> ', calc_test_error())
output_tweet_sample()
for e in range(0, NUM_EPOCHS):
    train_epoch(e)
    # save_path = saver.save(sess, CACHE_DIR + "/model/model.ckpt", global_step=e)
    # print("Model saved in file: %s" % save_path)
