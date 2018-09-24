import ArgumentParser
from NeuralNet import *
from Config import Config
import tflowtools as TFT
import sys
from CaseManager import *
import testing.DataLoader as DataLoader
import numpy as np
import mnist.mnist_basics as mnist
from Case import *


def to_one_hot_int_1d(length, number):
    zeros = np.zeros(length)
    zeros[number] = 1
    return zeros


def scale_255(number):
    return number / 255


def scale_array(array):
    out = []

    for i in array:
        out.append(scale_255(i))
    return out


if __name__ == '__main__':
    args = ArgumentParser.parseArgs()
    config = Config(args)
    caseManager = CaseManager(cfunc=lambda: config.src_function(*config.src_args),
                              vfrac=config.vfrac,
                              tfrac=config.tfrac,
                              case_fraction=config.case_fraction)

    # caseManager = CaseManager(cfunc=(lambda: TFT.gen_symvect_dataset(config.layer_sizes[0], count=2000)),
    #                           vfrac=config.vfrac,
    #                           tfrac=config.tfrac,
    #                           case_fraction=config.case_fraction)

    nn = NeuralNet(config, caseManager)
    print(nn)
    #exit(1)

    nn.do_training()

    nn.do_testing()

    # raw = mnist.load_flat_cases("all_flat_mnist_training_cases")
    # print(len(raw[0]))
    # print(len(raw[1]))
    # images = raw[0]
    # labels = raw[1]
    #
    # S = len(raw[0]) * 0.1
    # TeF = args.tfrac
    # VaF = args.vfrac
    #
    # training_set_size = round(S * (1 - (TeF + VaF)))
    # validation_set_size = round(S * VaF)
    # test_set_size = round(S * TeF)
    #
    # print(len(images[0]))
    # print(images[:1])
    # print(labels[:1])
    # labels_one_hot = [to_one_hot_int_1d(10, label) for label in labels[:1000]]
    # scaled_images = [scale_array(image) for image in images[:1000]]
    # print(scaled_images[0])
    # print(labels_one_hot[0])
    #
    # nn.do_training(scaled_images, labels_one_hot)
    #
    # labels_one_hot = [to_one_hot_int_1d(10, label) for label in labels[11:12]]
    # scaled_images = [scale_array(image) for image in images[11:12]]
    #
    # print(scaled_images)
    # print(labels_one_hot)
    # nn.do_testing(scaled_images, labels_one_hot)
