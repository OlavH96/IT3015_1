import sys

import matplotlib.pyplot as plt
import tensorflow as tf

import ArgumentParser
import testing.DataLoader as DataLoader
import tflowtools as TFT
import numpy as np
from Config import *


def showSomeImages(pairs, n):
    for i in range(0, n):
        pair = pairs[i]
        image = pair.image
        label = pair.label

        plt.imshow(image, cmap="gray")
        plt.title(label[0])
        plt.show()


# generates a weight variable of a given shape.
def generate_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# generates a bias variable of a given shape.
def generate_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def scaleNumber(number, start=0, end=1, max=255):
    return (number / max) * (end - start)


if __name__ == '__main__':

    # Handle Arguments
    args = ArgumentParser.parseArgs()
    print(args)
    config = Config(args)
    print(config.haf)
    print(config.optimizer)
    (layers, sizes) = ArgumentParser.handleNDIM(args.ndim)
    print("Layers: ", layers, ", Sizes: ", sizes)
    steps = args.steps
    learning_rate = args.lr
    (images, labels) = DataLoader.load('training')
    flat_labels = np.array([label[0] for label in labels])

    S = len(images)
    TeF = args.tfrac
    VaF = args.vfrac

    training_set_size = round(S * (1 - (TeF + VaF)))
    validation_set_size = round(S * VaF)
    test_set_size = round(S * TeF)

    (training_set, validation_set, test_set) = DataLoader.split(images, labels,
                                                                training_set_size,
                                                                validation_set_size,
                                                                test_set_size)

    yDim = len(images[0])
    xDim = len(images[0][0])

    # Construct model
    #  Input nodes
    x = tf.placeholder(dtype=tf.float32, shape=[None, xDim, yDim], name='image')  # Image data
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='label')  # Label
    images_flat = tf.layers.flatten(x)

    logits = tf.contrib.layers.fully_connected(inputs=images_flat,
                                               num_outputs=10,
                                               activation_fn=tf.nn.relu)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                         logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    correct_pred = tf.argmax(logits, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Start Training
    tf.set_random_seed(1234)
    sess = tf.Session()
    # TFT.viewprep(sess)

    sess.run(tf.global_variables_initializer())
    accuracy_data = []
    for i in range(steps):
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images, y: flat_labels})
        accuracy_data.append(accuracy_val)
        print("Accuracy: ", accuracy_val)
        if i % 10 == 0:
            print("Epoch: ", i)

    test_set_images = [t.image for t in test_set]
    test_set_labels = [t.label[0] for t in test_set]

    predicted = sess.run([correct_pred], feed_dict={x: test_set_images})[0]

    print("Predicted", predicted)
    print("Actual: ", test_set_labels)

    correct = 0
    total = test_set_size

    for i in range(0, len(predicted)):
        p = predicted[i]
        ac = test_set_labels[i]
        if p == ac:
            correct = correct + 1

    print("Total: ", total)
    print("Correct: ", correct)

    print("Accuracy on test set: ", correct / total)

    for i in range(0, 10):
        tpair = test_set[i]
        image = tpair.image
        actual_label = tpair.label
        pred = predicted[i]

        plt.imshow(image, cmap="gray")

        title = "Truth:        {0}\nPrediction: {1}".format(actual_label, pred)
        plt.title(title)
        plt.show()

    plt.plot(accuracy_data)
    plt.title(str(learning_rate))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()

    # TFT.fireup_tensorboard('probeview')
