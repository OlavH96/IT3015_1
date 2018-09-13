import ArgumentParser
from NeuralNet import *
from Config import Config
import tflowtools as TFT
import sys

if __name__ == '__main__':

    args = ArgumentParser.parseArgs()
    config = Config(args)
    nn = NeuralNet(config)

    print(nn)

    cases = TFT.gen_symvect_dataset(vlen=nn.input_layer_size, count=2000)
    data = [case[:-1] for case in cases]
    labels = [[case[2]] for case in cases]
    print(cases)
    print(data)
    print(labels)
    nn.do_training(data, labels)

    test_cases = TFT.gen_symvect_dataset(vlen=nn.input_layer_size, count=200)
    test_data = [case[:-1] for case in test_cases]
    test_labels = [case[2] for case in test_cases]
    print(test_data)
    print(test_labels)
    nn.do_testing(test_data, test_labels)