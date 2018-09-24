import tensorflow as tf
import tflowtools as TFT


class Config:
    def __init__(self, args):
        self.args = args
        print(args)

        (self.number_of_layers, self.layer_sizes) = handleNDIM(self.args.ndim)
        self.haf = parseHAF(self.args.haf)
        self.oaf = parseOAF(self.args.oaf)
        self.cf = parseCF(self.args.cf)
        self.lr = self.args.lr
        self.iwr_lower_bound, self.iwr_upper_bound = parseIWR(self.args.iwr)
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

        self.src_function, self.src_args = handleSrc(self.args.src)


def handleSrc(src):

    type = src[0]

    if type == "function":
        return handleSrcFunction(src)
    elif type == "file":
        return handleSrcFile(src)
    else:
        raise Exception("Invalid src type, must be 'function' or 'file'")


def handleSrcFile(src):
    return "NYI"

def handleSrcFunction(src):
    module_name = src[1]
    function_name = src[2]
    function_arguments = src[3:]

    module = __import__(module_name)  # import the module
    function = getattr(module, function_name)
    rest = list(map(lambda x: int(x), function_arguments))

    function(*rest)  # * unpacks list into arguments for the function

    print("SRC")
    print(type)
    print(module_name)
    print(function_name)
    print(function_arguments)

    return function, rest

def handleNDIM(ndim):
    print(ndim)

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
        "adagrad": tf.train.AdagradDAOptimizer
    }
    return options[optimizer] or options["gradientdescent"]


def parseMapLayers(map_layers):
    return "NYI"


def parseIWR(iwr):
    if isinstance(iwr, str):
        print("IWR is a string, panic!")
        return iwr
    if len(iwr) is not 2:
        raise Exception("IWR length must be 2")
    print("IWR is " + str(iwr))
    return iwr[0], iwr[1]


def parseMDend(mdend):
    return "NYI"


def parseDW(dw):
    return "NYI"


def parseDB(db):
    return "NYI"
