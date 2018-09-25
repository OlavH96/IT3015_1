import ArgumentParser
from CaseManager import CaseManager
from Config import Config
from NeuralNet import NeuralNet
import tflowtools as TFT
import mnist.mnist_basics as mnist

if __name__ == '__main__':

    args = ArgumentParser.parseArgs()
    config = Config(args)

    raw = mnist.load_flat_cases("all_flat_mnist_training_cases")
    raw_data = raw[0]
    raw_labels = raw[1]
    labels = [TFT.int_to_one_hot(l, 10) for l in raw_labels]

    print(len(raw_data))
    print(len(raw_data[0]))
    print(raw_labels[0])
    print(labels[0])
    data_set = raw_data[:round(len(raw_data) * 0.1)]  # 10% of images

    caseManager = CaseManager(cases=data_set, labels=labels)

    nn = NeuralNet(config, caseManager)
    nn.do_training()
    nn.do_testing(caseManager.get_testing_cases())
