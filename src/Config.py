import os

import tensorflow as tf
import tflowtools as TFT
import re as regex
import mnist.mnist_basics as mnist
import math
import importlib


class Config:
    def __string_array_to_type(self, array, type):
        to = lambda x: type(x)
        return [to(x) for x in array]

    def __init__(self, args):
        self.args = args

        (self.number_of_layers, self.layer_sizes) = handleNDIM(self.args.ndim)
        self.haf = parseHAF(self.args.haf)
        self.oaf = parseOAF(self.args.oaf)
        self.cf = parseCF(self.args.cf)
        self.lr = self.args.lr
        self.iwr_function = parseIWR(self.args.iwr)
        # self.iwr_lower_bound, self.iwr_upper_bound = parseIWR(self.args.iwr)
        self.optimizer = parseOptimizer(self.args.optimizer)
        self.src = self.args.src
        self.case_fraction = self.args.case_fraction
        self.vfrac = self.args.vfrac
        self.vint = self.args.vint
        self.tfrac = self.args.tfrac
        self.msize = self.args.msize
        self.mbsize = self.args.mbsize
        self.steps = self.args.steps
        self.map_layers = parseMapLayers(self.args.map_layers)
        self.mdend = parseMDend(self.args.mdend)
        self.dw = parseDW(self.args.dw)
        self.db = parseDB(self.args.db)
        self.src_function, self.src_args, self.src_file_path = handleSrc(self.args.src)
        self.one_hot_output = [parseArgType(x) for x in self.args.one_hot_output]

        self.scale_input = [parseArgType(x) for x in self.args.scale_input]
        self.topk = self.args.topk


def handleSrc(src):
    type = src[0]

    if type == "function":
        return handleSrcFunction(src)
    elif type == "file":
        return handleSrcFile(src)
    else:
        raise Exception("Invalid src type, must be 'function' or 'file'")


def handleSrcFile(src):
    type = src[0]
    file_path = src[1]
    module_name = src[2]
    function_name = src[3]
    function_args = src[4:]

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    args = list(map(lambda x: parseArgType(x), function_args))

    return function, args, file_path


def handleSrcFunction(src):
    module_name = src[1]
    function_name = src[2]
    function_arguments = src[3:]

    module = importlib.import_module(module_name)

    function = getattr(module, function_name)
    args = list(map(lambda x: parseArgType(x), function_arguments))

    # function(*args)  # * unpacks list into arguments for the function
    return function, args, None


def parseArgType(arg):
    if regex.match("^\d+$", arg):
        return int(arg)
    elif regex.match("^\d+\.\d*$", arg):
        return float(arg)
    elif regex.match("(True|False)", arg):
        return True if arg == "True" else False
    else:
        return str(arg)


def handleNDIM(ndim):
    if ndim is None:
        return
    dim = ndim[0]
    rest = len(ndim[0:-1])

    if dim is not rest:
        raise Exception("Network dimensions does not match: %i layers, got %i sizes" % (dim, rest))
    else:
        return dim, ndim[1:]


def parseHAF(haf):
    options = {
        "sigmoid": tf.sigmoid,
        "tanh": tf.tanh,
        "relu": tf.nn.relu,
        "softmax": tf.nn.softmax
    }
    return options[haf] or options["sigmoid"]


def parseOAF(oaf):
    return parseHAF(oaf)


def parseCF(cf):
    options = {
        "mse": tf.losses.mean_squared_error,
        "ce": tf.losses.sigmoid_cross_entropy
    }
    return options[cf] or options["mse"]


def parseOptimizer(optimizer):
    options = {
        "gradientdescent": tf.train.GradientDescentOptimizer,
        "adam": tf.train.AdamOptimizer,
        "rmsprop": tf.train.RMSPropOptimizer,
        "adagrad": tf.train.AdagradOptimizer
    }
    return options[optimizer] or options["gradientdescent"]


def parseIWR(iwr):
    if len(iwr) == 2:

        return lambda nodes: [float(iwr[0]), float(iwr[1])]
    elif len(iwr) == 1 and iwr[0] == "scaled":

        return lambda nodes: [-(1 / math.sqrt(nodes)), (1 / math.sqrt(nodes))]

    raise Exception("IWR length must be 2, or the word 'scaled'")


def parseMDend(mdend):
    toInt = lambda x: int(x)
    return [toInt(x) for x in mdend]


def parseDW(dw):
    toInt = lambda x: int(x)
    return [toInt(x) for x in dw]


def parseDB(db):
    toInt = lambda x: int(x)
    return [toInt(x) for x in db]


def parseMapLayers(map_layers):
    toInt = lambda x: int(x)
    return [toInt(x) for x in map_layers]
