import tflowtools as tft
import argparse
import matplotlib.pyplot as plt
import math
import ArgumentParser
import sys
import tensorflow as tf
import mnist.mnist_basics as mnist
import os
import sys
import random
import testing.DataLoader as DataLoader
import testing.NeuralNet as NeuralNet
from ImageWithLabel import *


def showSomeImages(pairs, n):
    for i in range(0, n):
        pair = pairs[i]
        image = pair.image
        label = pair.label

        plt.imshow(image, cmap="gray")
        plt.title(label[0])
        plt.show()


if __name__ == '__main__':

    sys.argv.append("-src")
    sys.argv.append(".")

    args = ArgumentParser.parseArgs()
    print(args)

    (layers, sizes) = ArgumentParser.handleNDIM(args.ndim)
    print(layers)
    print(sizes)

    haf = args.haf
    print(haf)

    src = args.src
    print(src)

    (images, labels) = DataLoader.load()
    S = len(images)
    TeF = args.tfrac
    VaF = args.vfrac

    training_set_size = S * (1 - (TeF + VaF))
    validation_set_size = S * VaF
    test_set_size = S * TeF

    training_set_size = round(0.04 * S)
    validation_set_size = round(0.01 * S)
    test_set_size = round(0.01 * S)

    print(training_set_size)
    print(validation_set_size)
    print(test_set_size)

    training_set = []
    validation_set = []
    test_set = []

    for i in range(0, training_set_size):
        pair = ImageWithLabel(images[i], labels[i])
        training_set.append(pair)

    for i in range(training_set_size, training_set_size + validation_set_size):
        pair = ImageWithLabel(images[i], labels[i])
        validation_set.append(pair)

    for i in range(training_set_size + validation_set_size, training_set_size + validation_set_size + test_set_size):
        pair = ImageWithLabel(images[i], labels[i])
        test_set.append(pair)

    # nnet = NeuralNet.NeuralNet(1)

    # showSomeImages(test_set, 10)

    yDim = len(images[0])
    xDim = len(images[0][0])

    x = tf.placeholder(dtype=tf.float32, shape=[None, xDim, yDim])  # Image data
    y = tf.placeholder(dtype=tf.int32, shape=[None])  # Label

    images_flat = tf.contrib.layers.flatten(x)  # Flatten 2d array to 1d

    logits = tf.contrib.layers.fully_connected(images_flat, 10, tf.nn.relu)  # 10 output nodes

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                         logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    correct_pred = tf.argmax(logits, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)

    tf.set_random_seed(1234)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images[i], y: labels[i]})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')
