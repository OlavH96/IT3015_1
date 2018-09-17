import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='Configure neural net')

    parser.add_argument('-ndim', dest='ndim', type=int, nargs="*", help='Network Dimensions', default=[2, 2, 1])
    parser.add_argument('-haf', dest='haf', type=str, help='Hidden Activation Function', default="sigmoid",
                        choices=["sigmoid", "tanh", "elu", "softmax", "softplus", "softsign", "relu", "relu6", "crelu", "relu_x"])
    parser.add_argument('-oaf', dest='oaf', type=str, help='Output Activation Function', default="sigmoid",
                        choices=["sigmoid", "tanh", "elu", "softmax","softplus", "softsign", "relu", "relu6", "crelu", "relu_x"])
    parser.add_argument('-cf', dest='cf', help='Cost Function', default="mse", choices=["mse", "ce"])
    parser.add_argument('-lr', dest='lr', help='Learning Rate', type=float, default=0.01)
    parser.add_argument('-iwr', dest='iwr', type=float, nargs="*", help='Initial Weight Range', default=[-0.1, 0.1])
    parser.add_argument('-optimizer', dest='optimizer', help='Optimizer', type=str, default="gradientdescent", choices=["gradientdescent", "adam", "rmsprop","adagrad"])
    parser.add_argument('-src', dest='src', help='Data Source', type=str)  # required=True
    parser.add_argument('-case_fraction', dest='case_fraction', help='Case Fraction', type=float, default=1)
    parser.add_argument('-vfrac', dest='vfrac', help='Validation Fraction', type=float, default=0.1)
    parser.add_argument('-vint', dest='vint', help='Validation Interval', type=float, default=1)
    parser.add_argument('-tfrac', dest='tfrac', help='Test Fraction', type=float, default=0.1)
    parser.add_argument('-msize', dest='msize', help='Minibatch Size', type=int, default=100)
    parser.add_argument('-mbsize', dest='mbsize', help='Map Batch Size', type=int, default=0)
    parser.add_argument('-steps', dest='steps', help='Minibatch Steps', type=int, default=100)
    parser.add_argument('-map_layers', dest='map_layers', help='Map Layers', type=int, default=0)
    parser.add_argument('-mdend', dest='mdend', help='Map Dendrograms', type=list, default=[])
    parser.add_argument('-dw', dest='dw', help='Display Weights', type=list, default=[])
    parser.add_argument('-db', dest='db', help='Display Biases', type=list, default=[])

    args = parser.parse_args()

    return args


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
