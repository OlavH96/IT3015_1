import ArgumentParser
from NeuralNet import *
from Config import Config
import tflowtools as TFT
import sys
from CaseManager import *
import numpy as np
import mnist.mnist_basics as mnist
from Case import *
import CSVReader

if __name__ == '__main__':
    args = ArgumentParser.parseArgs()
    config = Config(args)
    caseManager = CaseManager(config,
                              cfunc=lambda: config.src_function(*config.src_args),  # * unpacks list arguments
                              vfrac=config.vfrac,
                              tfrac=config.tfrac,
                              case_fraction=config.case_fraction,
                              src_function=config.src_function,
                              src_args=config.src_args,
                              src_path=config.src_file_path)

    nn = NeuralNet(config, caseManager)

    nn.do_training()

    # nn.do_testing()
    # TFT.fireup_tensorboard('probeview')
