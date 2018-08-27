import mnist.mnist_basics as mnist
from ImageWithLabel import *

def load(dataset="training"):
    raw = mnist.load_mnist(dataset)
    images = raw[0]
    labels = raw[1]

    return images, labels

def split(data, labels, training_size, validation_size, test_size):

    training_set = []
    validation_set = []
    test_set = []

    for i in range(0, training_size):
        pair = ImageWithLabel(data[i], labels[i])
        training_set.append(pair)

    for i in range(training_size, training_size + validation_size):
        pair = ImageWithLabel(data[i], labels[i])
        validation_set.append(pair)

    for i in range(training_size + validation_size, training_size + validation_size + test_size):
        pair = ImageWithLabel(data[i], labels[i])
        test_set.append(pair)
    return training_set, validation_set, test_set
