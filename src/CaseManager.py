import numpy as np
from Case import *


class CaseManager:

    def __init__(self, cfunc=None, cases=None, labels=None, vfrac=0.1, tfrac=0.1, case_fraction=1):
        self.casefunc = cfunc
        self.case_fraction = case_fraction
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.labels = labels
        if cases:
            print("has cases")
            self.cases = cases
        else:
            print("Generating cases")
            self.generate_cases()

        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        # Handle case fraction
        ca = ca[:round(len(ca)*self.case_fraction)]
        self.cases = self.cases[:round(len(self.cases)*self.case_fraction)]

        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[:separator1]
        self.training_cases = [Case(input=case[:-1], target=[case[-1]]) for case in self.training_cases]
        self.validation_cases = ca[separator1:separator2]
        self.validation_cases = [Case(input=case[:-1], target=[case[-1]]) for case in self.validation_cases]
        self.testing_cases = ca[separator2:]
        self.testing_cases = [Case(input=case[:-1], target=[case[-1]]) for case in self.testing_cases]

        if self.labels:
            self.training_cases = [Case(input=self.training_cases[i], target=self.labels[i]) for i in range(len(self.training_cases))]
            self.training_cases = [Case(input=case[:-1], target=[case[-1]]) for case in self.training_cases]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases
