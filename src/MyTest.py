import ArgumentParser
from NeuralNet import *
from Config import Config
import tflowtools as TFT
import sys
from CaseManager import *

if __name__ == '__main__':
    args = ArgumentParser.parseArgs()
    config = Config(args)

    caseManager = CaseManager(cfunc=(lambda: TFT.gen_symvect_dataset(config.layer_sizes[0], count=2000)), vfrac=0,
                              tfrac=0.1)
    nn = NeuralNet(config, caseManager)
    print(nn)

    cases = TFT.gen_symvect_dataset(vlen=nn.input_layer_size, count=500)
    data = [case[:-1] for case in cases]
    labels = [[case[2]] for case in cases]

    nn.do_training(data, labels)

    test_cases = TFT.gen_symvect_dataset(vlen=nn.input_layer_size, count=5000)
    test_data = [case[:-1] for case in test_cases]
    test_labels = [case[2] for case in test_cases]

    nn.do_testing(test_data, test_labels)
