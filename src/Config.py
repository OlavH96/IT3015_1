import tensorflow as tf


class Config:
    def __init__(self, args):
        self.args = args
        print(args)

        self.ndim = handleNDIM(self.args.ndim)
        self.haf = parseHAF(self.args.haf)
        self.oaf = parseOAF(self.args.oaf)
        self.cf = self.args.cf
        self.lr = self.args.lr
        self.iwr = self.args.iwr
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
        "default": tf.sigmoid
    }
    return options[haf] or options["default"]


def parseOAF(oaf):
    return parseHAF(oaf)


def parseOptimizer(optimizer):
    options = {
        "gradientdecent": tf.train.GradientDescentOptimizer,
        "adam": tf.train.AdamOptimizer,
        "default": tf.train.GradientDescentOptimizer
    }
    return options[optimizer] or options["default"]


def parseMapLayers(map_layers):
    return "NYI"


def parseMDend(mdend):
    return "NYI"


def parseDW(dw):
    return "NYI"


def parseDB(db):
    return "NYI"
