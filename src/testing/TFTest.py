
import tensorflow as tf
import os
import sys
import skimage
from skimage import data
import matplotlib.pyplot as plt
from skimage import transform

def test():
    print("Test")
    path = os.path.join(".", "testing", "data")
    files = os.listdir(path)

    images = []
    for file in files:
        images.append(skimage.data.imread(os.path.join(path, file)))

    traffic_signs = [1, 2, 3, 4]
    images28 = [transform.resize(image, (28, 28)) for image in images]

    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images28[traffic_signs[i]])
        plt.subplots_adjust(wspace=0.5)

    plt.show()


