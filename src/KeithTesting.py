import tensorflow as tf
import numpy as np
import tflowtools as tft
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 3 input nodes
    # 1 output node
    # 2 hidden nodes i ett layer

    input_size = 11
    hidden_nodes = 5
    output_size = 1
    data_length = 1000
    learning_rate = 1
    steps = 500

    raw_data = tft.gen_symvect_dataset(input_size, data_length)

    data = [data[:input_size] for data in raw_data]
    labels = [[data[input_size]] for data in raw_data]
    print(raw_data)
    print(data)
    print(labels)
    labels = np.array(labels)

    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name='y')

    w = tf.Variable(tf.random_normal([input_size, hidden_nodes], stddev=0.03), name='w1')
    b = tf.Variable(tf.random_normal([hidden_nodes]), name='b1')

    hidden_output = tf.tanh(tf.matmul(x, w) + b)

    w2 = tf.Variable(tf.random_normal([hidden_nodes, output_size], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([output_size]), name='b2')

    sig = tf.sigmoid

    y_est = sig(tf.matmul(hidden_output, w2) + b2)

    error = tf.reduce_mean(tf.square(y_est - y))  # MSE

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training = optimizer.minimize(error)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # tft.viewprep(sess)
    # tft.fireup_tensorboard(logdir='probeview')

    errors = []
    for i in range(steps):
        _, err = sess.run([training, error], feed_dict={x: data, y: labels})
        errors.append(err)

    plt.scatter(range(steps), errors, s=5)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()

    test_data_size = int(data_length/5)

    raw_test_data = tft.gen_symvect_dataset(input_size, test_data_size)

    test_data = [data[:input_size] for data in raw_test_data]
    test_labels = [[data[input_size]] for data in raw_test_data]
    res = sess.run(y_est, feed_dict={x: test_data})
    print(res)

    correct = 0
    for i in range(test_data_size):
        data = test_data[i]
        label = test_labels[i][0]
        est = res[i][0]

        # print(data)
        # print(label)
        # print(est)

        if label == 1 and est > 0.5:
            correct += 1
        if label == 0 and est < 0.5:
            correct += 1

    print(correct, " / ", test_data_size, " correct")
    print((correct/test_data_size) * 100, " % correct")

    sess.close()
