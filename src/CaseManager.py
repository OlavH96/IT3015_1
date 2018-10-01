import numpy as np
from Case import *
import tflowtools as tft


def unpack(array):
    if type(array) is not list: return [array]  # pack into array

    if type(array) is list and type(array[0]) is not list:  # is a single array
        return array
    return unpack(array[0])  # unpack to single array


def scaleNumber(number, start=0, end=1, max=255):
    return (number / max) * (end - start)


def scale(number, min, max):
    return (number / (max - min))


def scaleArray(array, min, max):
    return [scale(n, min, max) for n in array]


class CaseManager:

    def __init__(self, config, cfunc=None, cases=None, labels=None, vfrac=0.1, tfrac=0.1, case_fraction=1,
                 src_function=None,
                 src_args=None,
                 src_path=None,
                 ):
        self.config = config
        self.casefunc = cfunc
        self.case_fraction = case_fraction
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.labels = labels
        self.src_path = src_path
        if cases:
            print("has cases")
            self.cases = cases
        elif src_path:  # has file to read
            res = src_function(*src_args, src_path)
            print(res)
        else:
            print("Generating cases")
            self.generate_cases()

        self.organize_cases()
        print(len(self.cases))

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases

        # Handle case fraction
        ca = ca[:round(len(ca) * self.case_fraction)]
        ca = ca.tolist()
        self.cases = self.cases[:round(len(self.cases) * self.case_fraction)]

        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[:separator1]
        self.training_cases = self.createCases(self.training_cases)

        self.validation_cases = ca[separator1:separator2]
        self.validation_cases = self.createCases(self.validation_cases)

        self.testing_cases = ca[separator2:]
        self.testing_cases = self.createCases(self.testing_cases)

        if self.config.scale_input:
            print("Scale input to ", self.config.scale_input)

            for x in self.training_cases:
                x.input = scaleArray(x.input, self.config.scale_input[0], self.config.scale_input[1])
            for x in self.validation_cases:
                x.input = scaleArray(x.input, self.config.scale_input[0], self.config.scale_input[1])
            for x in self.testing_cases:
                x.input = scaleArray(x.input, self.config.scale_input[0], self.config.scale_input[1])

        if self.labels:
            self.training_cases = [Case(input=self.training_cases[i], target=self.labels[i]) for i in
                                   range(len(self.training_cases))]
            self.training_cases = [Case(input=case[:-1], target=[case[-1]]) for case in self.training_cases]

    def createCases(self, list):
        if self.config.one_hot_output and self.config.one_hot_output[0]:
            return [Case(input=unpack(case[:-1]),
                         target=unpack(tft.int_to_one_hot(case[-1], size=self.config.one_hot_output[-1]))) for case in
                    list]
        else:
            return [Case(input=unpack(case[:-1]), target=unpack(case[-1])) for case in list]

    def get_training_cases(self):
        return self.training_cases

    def get_validation_cases(self):
        return self.validation_cases

    def get_testing_cases(self):
        return self.testing_cases
