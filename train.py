import json
import math
import os

import numpy as np
import tensorflow as tf

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'
BATCH_SIZE = 64

with open(CACHE_DIR + '/settings.json') as file:
    settings = json.load(file)

max_steps = settings['maxSteps']

x = tf.placeholder(dtype=tf.int32, shape=[None, max_steps, 2], name='x')
y = tf.placeholder(dtype=tf.int32, shape=[None, max_steps, 2], name='y')
length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='length')

features = np.load(CACHE_DIR + '/features.npy')
labels = np.load(CACHE_DIR + '/labels.npy')
lengths = np.load(CACHE_DIR + '/lengths.npy')

sess = tf.Session()

sess.run(tf.global_variables_initializer())

num_examples = features.shape[0]
num_batches = math.ceil(num_examples / BATCH_SIZE)

for batch in range(num_batches):
    start = batch * BATCH_SIZE
    end = (batch + 1) * BATCH_SIZE
    sess.run(length, feed_dict={x: features[start:end, :, :], y: labels[start:end, :, :], length: lengths[start:end, :]})
