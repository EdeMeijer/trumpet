import random

import numpy as np
import tensorflow as tf

# Max length can go over the normal tweet length limit to give the model some slack
MAX_SAMPLE_LENGTH = 200


class TweetSampler:
    def __init__(self, session, model, temperature=1.0):
        self.session = session
        self.model = model
        self.predictions_flat = tf.nn.softmax(model.logits / temperature, 1)

    def sample(self):
        # Start with the start symbol, which has label num_chars
        features = [len(self.model.chars)]
        tweet = ''

        for i in range(MAX_SAMPLE_LENGTH):
            next_class = self.sample_next_class(features[-self.model.max_steps:])
            features.append(next_class)
            if next_class == len(self.model.chars):
                break
            tweet += self.model.chars[next_class]

        return tweet.strip()

    def sample_next_class(self, classes):
        sample_input = np.zeros([1, self.model.max_steps, 1])
        sample_input[:, :len(classes)] = np.array(classes).reshape([1, len(classes), 1])

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