import json
import math
import os

import tensorflow as tf

import data_set
from model import Model
from tweet_sampler import TweetSampler

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'
BATCH_SIZE = 64
NUM_EPOCHS = 200

test_data, train_data = data_set.load().split_test_train()

with open(CACHE_DIR + '/settings.json') as file:
    settings = json.load(file)

model = Model(
    settings['chars'],
    settings['maxSteps'],
    lstm_units=500,
    l1_units=400,
    l2_units=300,
    l2=0.00005
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sampler = TweetSampler(sess, model, temperature=0.8)


def output_tweet_sample():
    print('Sampling tweet....\n\n---------------------------------------')
    print(sampler.sample())
    print('---------------------------------------\n')


def process_data(data, ops):
    num_examples = data.features.shape[0]
    num_batches = math.ceil(num_examples / BATCH_SIZE)
    for batch in range(num_batches):
        start = batch * BATCH_SIZE
        end = (batch + 1) * BATCH_SIZE
        yield sess.run(
            ops,
            feed_dict={
                model.features: data.features[start:end],
                model.labels: data.labels[start:end],
                model.mask: data.mask[start:end]
            }
        )


def calc_test_error():
    total_err = 0
    num_batches = 0
    for err in process_data(test_data, ops=model.loss):
        total_err += err
        num_batches += 1
    return total_err / num_batches


def train_epoch(epoch):
    num_batches = 0
    total_err = 0
    total_l2 = 0

    ops = [model.train_op, model.loss, model.l2_loss]
    for _, err, l2 in process_data(train_data, ops=ops):
        total_err += err
        total_l2 += l2
        num_batches += 1

    print('EPOCH {}: train = {:.5}, test = {:.5}, L2 = {:.5}'.format(
        epoch, total_err / num_batches, calc_test_error(), total_l2 / num_batches))
    output_tweet_sample()


saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)
output_tweet_sample()
for epoch in range(0, NUM_EPOCHS):
    train_epoch(epoch)
    save_path = saver.save(sess, CACHE_DIR + "/model/model.ckpt", global_step=epoch)
    print("Model saved in file: %s" % save_path)
