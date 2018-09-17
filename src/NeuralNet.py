import tensorflow as tf
import numpy as np
from Layer import *
from CaseManager import *
import matplotlib.pyplot as plt
import tflowtools as TFT

class NeuralNet:

    def __init__(self, config, case_manager):
        self.config = config
        self.case_manager = case_manager
        self.layers = []

        self.learning_rate = config.lr
        self.number_of_layers = config.number_of_layers
        self.layer_sizes = config.layer_sizes
        self.input_layer_size = self.layer_sizes[0]
        self.output_layer_size = self.layer_sizes[-1]
        self.hidden_layer_sizes = self.layer_sizes[1:-1] if len(self.layer_sizes) > 2 else []
        self.oaf = config.oaf
        self.haf = config.haf
        self.optimizer = config.optimizer
        self.steps = config.steps

        self.iwr_lower_bound, self.iwr_upper_bound = config.iwr_lower_bound, config.iwr_upper_bound

        print(self.layers)
        print(self.layer_sizes)
        print(self.oaf)
        print(self.haf)
        self.build()

    def build(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(dtype=tf.float64, shape=[None, self.input_layer_size],
                                    name='input_layer')  # Image data

        invar = self.input
        input_size = self.input_layer_size

        for i, outsize in enumerate(self.layer_sizes[1:]):
            layer = Layer(net=self, index=i, input=invar, input_size=input_size, output_size=outsize,
                          activation_function=self.oaf if i is self.number_of_layers - 2 else self.haf, iwr_lower_bound=self.config.iwr_lower_bound, iwr_upper_bound=self.config.iwr_upper_bound)
            invar = layer.output
            input_size = layer.output_size

        self.output = layer.output  # Output of last module is output of whole network

        self.target = tf.placeholder(dtype=tf.float64, shape=[None, self.output_layer_size],
                                     name='output_layer')  # Image data

        self.configure_training()

    def configure_training(self):

        #self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        # self.error = tf.losses.mean_squared_error(self.target, self.output)
        self.error = self.config.cf(self.target, self.output)
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Optimizer')

    def do_training(self, cases, labels):

        sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())

        # TFT.viewprep(sess)
        # t_cases = self.case_manager.get_training_cases()
        # print(t_cases)

        feeder = {self.input: cases, self.target: labels}
        errors = []
        for i in range(self.steps):

            # For minibatch here
            _, res = sess.run([self.trainer, self.error], feed_dict=feeder)
            errors.append(res)
            if i % (self.steps / 10) == (self.steps / 10) - 1:
                print("Cost: "+str(res))
            # Consider validation testing here or something

        # TFT.fireup_tensorboard(logdir='probeview')

        plt.scatter(range(self.steps), errors, s=5)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()
        print(res)

    def do_testing(self, test_cases, test_labels):

        feeder = {self.input: test_cases, self.target: test_labels}

        sess = self.sess

        # correct_pred = tf.nn.in_top_k(tf.cast(self.output, tf.float32), tf.cast(self.target, tf.int32), 1)
        # sess.run(tf.global_variables_initializer())
        res = sess.run(self.output, feed_dict=feeder)
        # print(res)
        # return ""
        # TFT.viewprep(sess)
        # TFT.fireup_tensorboard('probeview')
        correct = 0
        for i in range(len(test_cases)):
            data = test_cases[i]
            label = test_labels[i][0]
            est = res[i][0]

            # print(data)
            # print(label)
            # print(est)

            if label == 1 and est > 0.5:
                correct += 1
            if label == 0 and est < 0.5:
                correct += 1

        print(correct, " / ", len(test_cases), " correct")
        print((correct / len(test_cases)) * 100, " % correct")

    def add_layer(self, layer):
        self.layers.append(layer)

    def __str__(self):
        out = "Net: layers=" + str(self.number_of_layers) + "\n"
        out += "Optimizer=" + str(self.optimizer) + ", learning_rate=" + str(self.learning_rate) + "\n"
        out += "CostFunction="+str(self.error)+"\n"
        out += str(self.input) + "\n"

        for layer in self.layers:
            out += str(layer) + "\n"

        return out
