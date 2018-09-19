import numpy as np
import matplotlib.pyplot as plt


def scatter(data_array, names):
    s = 1
    for data in data_array:
        d = [dat[1] for dat in data]
        points = [dat[0] for dat in data]
        # print(data)
        plt.scatter(points, d, s=s, label=names.pop(0))
        s += 5

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()


def line(data_array, names):
    for data in data_array:
        d = [dat[1] for dat in data]
        points = [dat[0] for dat in data]
        # print(data)
        plt.plot(points, d, label=names.pop(0))

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
