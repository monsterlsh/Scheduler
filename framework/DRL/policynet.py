import tensorflow as tf


class PolicyNet(tf.keras.Model):
    name = 'PolicyNet'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)
