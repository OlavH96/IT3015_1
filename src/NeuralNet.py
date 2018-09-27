from Layer import *
import math
from Case import *
import Plots
import tflowtools as TFT
import random


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
        self.grabbed_weigths_history = []
        self.grabbed_biases_history = []

        self.build()
        self.grabVars = [self.layers[i].weights for i in config.dw] + [self.layers[i].biases for i in config.db]

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

        self.error = self.config.cf(self.target, self.output)
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Optimizer')

    def do_training(self):
        case_list = self.case_manager.get_training_cases()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

        minibatch_size = self.minibatch_size
        number_of_cases = len(case_list)
        number_of_minibatches = math.ceil(number_of_cases / minibatch_size)

        # feeder = {self.input: case_list, self.target: labels}

        for step in range(self.steps):

            np.random.shuffle(case_list)  # Select random cases for this minibatch
            minibatch = case_list[:minibatch_size]  # if none, you just get the whole vector, if too large, you also just get the whole vector
            inputs = [case.input for case in minibatch]
            targets = [case.target for case in minibatch]

            m_feeder = {self.input: inputs, self.target: targets}

            _, res, grabbed = sess.run([self.trainer, self.error, self.grabVars],
                                       feed_dict=m_feeder)

            self.training_error_history.append((step, res))
            self.grabbed_weigths_history.append(grabbed)
            self.consider_validation_testing(step, sess)
            if step % (self.steps / 10) == 0:
                print(str((step / self.steps) * 100) + "% done, Cost: " + str(res))
                print("Validation Error: ", self.validation_error_history[-1][1])
            # Consider validation testing here or something

        # TFT.fireup_tensorboard(logdir='probeview')

        # Plots.line([errors, self.validation_error_history])
        print("Finished Training")
        print("Final training Error: " + str(self.training_error_history[-1][1]))
        print("Final validation Error: " + str(self.validation_error_history[-1][1]))
        Plots.scatter([self.training_error_history, self.validation_error_history],
                      ["Training Error", "Validation Error"])
        # Plots.plotWeights([self.grabbed_weigths_history])
        TFT.viewprep(sess)

        # Plots.line([errors, self.validation_error_history], ["Training Error", "Validation Error"])

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

    def do_testing(self, case_list=None, scenario="testing", printResult=False):

        if scenario == "testing" and not case_list:
            case_list = self.case_manager.get_testing_cases()

        inputs = [case.input for case in case_list]
        targets = [case.target for case in case_list]
        pred_targets = [t[0] for t in targets]
        sess = self.sess
        r = random.randint(0, len(targets)-1)
        # print(inputs[r])
        # print(targets[r])

        feeder = {self.input: inputs, self.target: targets}

        if self.output_layer_size == 1:  # one outout node, special case

            correct_pred = tf.equal(tf.round(self.output), targets)
            num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        else:
            correct_pred = tf.nn.in_top_k(tf.cast(self.output, tf.float32), tf.cast(pred_targets, tf.int32), 1)
            num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        out, res = sess.run([self.output, num_correct], feed_dict=feeder)
        # print(out[r])
        if scenario is not "validation":
            print(res, " / ", len(case_list), " correct")
            print((res / len(case_list)) * 100, " % correct")
            print(1 - (res / len(case_list)), " error")

        return 1 - (res / len(case_list))

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
