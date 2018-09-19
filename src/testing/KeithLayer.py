import tensorflow as tf
import numpy as np


class Layer:

    def __init__(self, ann, index, invariable, insize, outsize):
        self.ann = ann
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize  # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.build()

    def build(self):
        mona = self.name
        n = self.outsize
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize, n)),
                                   name=mona + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=mona + '-bias', trainable=True)  # First bias vector
        self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        self.ann.add_module(self)

    def getvar(self, type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self, type, spec):
        var = self.getvar(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/', var)
