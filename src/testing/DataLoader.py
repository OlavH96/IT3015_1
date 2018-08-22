import mnist.mnist_basics as mnist


def load():
    raw = mnist.load_mnist(dataset="training")
    images = raw[0]
    labels = raw[1]

    return images, labels
