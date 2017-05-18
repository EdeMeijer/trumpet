import random

import numpy as np
import tensorflow as tf

MAX_SAMPLE_LENGTH = 200


class TweetSampler:
    def __init__(self, session, model, temperature=1.0):
        self.session = session
        self.model = model
        self.predictions_flat = tf.nn.softmax(model.out_logits / temperature, 1)

    def sample(self):
        # Start with the start symbol, which has label num_chars
        features = [len(self.model.chars)]
        tweet = ''

        for i in range(MAX_SAMPLE_LENGTH):
            next_class = self.sample_next_class(features[-self.model.max_steps:])
            if next_class == len(self.model.chars):
                break
            features.append(next_class)
            tweet += self.model.chars[next_class]

        return tweet.strip()

    def sample_next_class(self, classes):
        sample_input = np.zeros([1, self.model.max_steps])
        sample_input[:, :len(classes)] = classes

        predictions = self.session.run(
            self.predictions_flat,
            feed_dict={
                self.model.features: sample_input,
            }
        )
        probabilities = predictions[len(classes) - 1]
        rnd = random.random()
        accum = 0
        for idx in range(len(probabilities)):
            accum += probabilities[idx]
            if accum >= rnd:
                return idx
        return np.argmax(classes)
