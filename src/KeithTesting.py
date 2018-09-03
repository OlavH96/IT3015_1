import tensorflow as tf
import numpy as np
import tflowtools as tft

if __name__ == '__main__':

    # 3 input nodes
    # 1 output node
    # 2 hidden nodes i ett layer
    #
    #
    #
    #
    #
    #

    input_size = 101
    output_size = 1
    data_length = 100
    learning_rate = 0.001
    steps = 100
    hidden_nodes = 3

    raw_data = tft.gen_symvect_dataset(input_size, data_length)

    data = [data[:input_size] for data in raw_data]
    labels = [[data[input_size]] for data in raw_data]
    print(raw_data)
    print(data)
    print(labels)

    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='x')

    w1 = tf.Variable(tf.random_normal([input_size, hidden_nodes], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([hidden_nodes], stddev=0.03), name='b1')

    w2 = tf.Variable(tf.random_normal([hidden_nodes, output_size], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([output_size]), name='b2')

    hidden_out = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

    y = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, w2), b2))

    labels = np.array(labels)
    error = tf.reduce_mean(tf.square(labels - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training = optimizer.minimize(error)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(steps):
        _, acc = sess.run([training, y], feed_dict={x: data, y: labels})
        if i % 10 == 9:
            print("Accuracy")
            print(acc)

    raw_test_data = tft.gen_symvect_dataset(input_size, 100)

    test_data = [data[:input_size] for data in raw_test_data]
    test_labels = [[data[input_size]] for data in raw_test_data]

    res = sess.run(y, feed_dict={x: test_data})
    print(res)
