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

    nnet = NeuralNet.NeuralNet(1)


    n = 10

    for i in range(0, n):
        image = images[i]
        label = labels[i]

        plt.imshow(image, cmap="gray")
        plt.title(label[0])
        plt.show()

    yDim = len(images[0])
    xDim = len(images[0][0])

    x = tf.placeholder(dtype=tf.float32, shape=[None, xDim, yDim])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    images_flat = tf.contrib.layers.flatten(x)

    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                         logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correct_pred = tf.argmax(logits, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.set_random_seed(1234)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            _, loss_value = sess.run([train_op, loss], feed_dict={x: images, y: labels})
            if i % 10 == 0:
                print("Loss: ", loss)
        sess.close()
