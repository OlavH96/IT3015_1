import tensorflow as tf
import numpy as np


class Layer:

    def __init__(self, net, index, input, input_size, output_size, activation_function):
        self.net = net
        self.index = index
        self.input = input
        self.input_size = input_size
        self.output_size = output_size
        self.name = "Layer-" + str(index)
        self.activation_function = activation_function

        self.build()

    def build(self):
        n = self.output_size

        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.input_size, n)),
                                   name=self.name + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=self.name + '-bias', trainable=True)  # First bias vector
        self.output = self.activation_function(tf.matmul(self.input, self.weights) + self.biases,
                                               name=self.name + '-out')

        self.net.add_layer(self)

    def __str__(self):
        return "Index=" + str(self.index) + ", ActivationFunction=" + str(self.activation_function) +", input_size="+str(self.input_size)+", output="+str(self.output_size)
