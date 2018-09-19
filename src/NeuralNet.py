from Layer import *
import math
from Case import *
import Plots


class NeuralNet:

    def __init__(self, config, case_manager):
        self.config = config
        self.case_manager = case_manager
        self.layers = []

        self.learning_rate = config.lr
        self.number_of_layers = config.number_of_layers
        self.layer_sizes = config.layer_sizes
        self.input_layer_size = self.layer_sizes[0]
        self.output_layer_size = self.layer_sizes[-1]
        self.hidden_layer_sizes = self.layer_sizes[1:-1] if len(self.layer_sizes) > 2 else []
        self.oaf = config.oaf
        self.haf = config.haf
        self.optimizer = config.optimizer
        self.steps = config.steps

        self.iwr_lower_bound, self.iwr_upper_bound = config.iwr_lower_bound, config.iwr_upper_bound

        self.minibatch_size = config.msize

        self.training_error_history = []
        self.validation_error_history = []

        print(self.layers)
        print(self.layer_sizes)
        print(self.oaf)
        print(self.haf)
        self.build()

    def build(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(dtype=tf.float64, shape=[None, self.input_layer_size],
                                    name='input_layer')  # Image data

        invar = self.input
        input_size = self.input_layer_size

        for i, outsize in enumerate(self.layer_sizes[1:]):
            layer = Layer(net=self, index=i, input=invar, input_size=input_size, output_size=outsize,
                          activation_function=self.oaf if i is self.number_of_layers - 2 else self.haf,
                          iwr_lower_bound=self.config.iwr_lower_bound, iwr_upper_bound=self.config.iwr_upper_bound)
            invar = layer.output
            input_size = layer.output_size

        self.output = layer.output  # Output of last module is output of whole network

        self.target = tf.placeholder(dtype=tf.float64, shape=[None, self.output_layer_size],
                                     name='output_layer')  # Image data

        self.configure_training()

    def configure_training(self):

        # self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        # self.error = tf.losses.mean_squared_error(self.target, self.output)
        self.error = self.config.cf(self.target, self.output)
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Optimizer')

    def do_training(self):
        case_list = self.case_manager.get_training_cases()
        sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())

        minibatch_size = self.minibatch_size
        number_of_cases = len(case_list)
        number_of_minibatches = math.ceil(number_of_cases / minibatch_size)

        # feeder = {self.input: case_list, self.target: labels}
        errors = []
        for step in range(self.steps):

            # For minibatch here
            for cstart in range(0, number_of_cases, minibatch_size):  # Loop through cases, one minibatch at a time.
                cend = min(number_of_cases, cstart + minibatch_size)
                minibatch = case_list[cstart:cend]

                inputs = [case.input for case in minibatch]
                targets = [case.target for case in minibatch]

                m_feeder = {self.input: inputs, self.target: targets}

                _, res = sess.run([self.trainer, self.error], feed_dict=m_feeder)
                errors.append((step, res))
            self.consider_validation_testing(step, sess)
            if step % (self.steps / 10) == 0:
                print(str((step / self.steps)*100)+"% done, Cost: " + str(res))
            # Consider validation testing here or something

        # TFT.fireup_tensorboard(logdir='probeview')

        # Plots.line([errors, self.validation_error_history])
        print("Finished Training")
        print("Final training Error: "+ str(errors[-1][1]))
        print("Final validation Error: "+ str(self.validation_error_history[-1][1]))
        Plots.scatter([errors, self.validation_error_history], ["Training Error", "Validation Error"])

    def should_run_validation_test(self, step):

        vint = self.config.vint
        return vint and (step % vint == 0)

    def consider_validation_testing(self, step, sess):

        if self.should_run_validation_test(step):
            case_list = self.case_manager.get_validation_cases()
            # case_list = [Case(input=case[:-1], target=[case[-1]]) for case in cases]

            error = self.do_testing(case_list, scenario="validation")
            # print(step, ", error=", error)
            self.validation_error_history.append((step, error))

    def do_testing(self, case_list, scenario="testing"):

        inputs = [case.input for case in case_list]
        targets = [case.target for case in case_list]
        targets_unpacked = [t[0] for t in targets]

        feeder = {self.input: inputs, self.target: targets}

        sess = self.sess

        correct_pred = tf.round(self.output)
        # correct_pred = tf.nn.in_top_k(tf.cast(self.output, tf.float32), tf.cast(targets_unpacked, tf.int32), 1)
        # sess.run(tf.global_variables_initializer())
        # sum = tf.reduce_sum(tf.equal(correct_pred, targets))
        res = sess.run(correct_pred, feed_dict=feeder)
        # print(res)
        # return ""
        # TFT.viewprep(sess)
        # TFT.fireup_tensorboard('probeview')
        correct = 0
        for i in range(len(case_list)):
            label = case_list[i].target[0]
            est = res[i][0]

            if label == est:
                correct += 1
        if scenario is not "validation":

            print(correct, " / ", len(case_list), " correct")
            print((correct / len(case_list)) * 100, " % correct")
            print(1-(correct / len(case_list)), " error")

        return 1 - (correct / len(case_list))

    def add_layer(self, layer):
        self.layers.append(layer)

    def __str__(self):
        out = "Net: layers=" + str(self.number_of_layers) + "\n"
        out += "Optimizer=" + str(self.optimizer) + ", learning_rate=" + str(self.learning_rate) + "\n"
        out += "CostFunction=" + str(self.error) + "\n"
        out += str(self.input) + "\n"

        for layer in self.layers:
            out += str(layer) + "\n"

        return out
