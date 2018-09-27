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
    caseManager = CaseManager(cfunc=lambda: config.src_function(*config.src_args),  # * unpacks list arguments
                              vfrac=config.vfrac,
                              tfrac=config.tfrac,
                              case_fraction=config.case_fraction,
                              src_function=config.src_function,
                              src_args=config.src_args,
                              src_path=config.src_file_path)

    nn = NeuralNet(config, caseManager)
    print(nn)

    # cases = caseManager.get_validation_cases()
    # for c in cases:
    #     print(c.input)
    #     print(c.target)

    nn.do_training()

    nn.do_testing()
    # TFT.fireup_tensorboard('probeview')

    # vect = [1, 0, 0, 0, 0, 0, 0, 0]
    # nn.do_testing(caseManager.get_validation_cases()[:10], printResult=True)
