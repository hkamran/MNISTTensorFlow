from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

OPTIONS = {}

def train():

    mnist = input_data.read_data_sets("./MNIST-data/",
                                      one_hot=True)

    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x-input")
        _y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="y-input")

    with tf.name_scope("input_reshape"):
        x_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])
        tf.summary.image(name="input", tensor=x_reshaped, max_outputs=10)

    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(
             inputs=x_reshaped,
             filters=32,
             strides=1,
             use_bias=True,
             padding="SAME",
             activation=tf.nn.relu,
             kernel_size=[5, 5],
             name="conv1")
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.name_scope("conv1_visual_weights"):
        conv1_w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
        conv1_w_images = tf.unstack(value=conv1_w, axis=3)
        tf.summary.image(name="conv1_weights", tensor=conv1_w_images, max_outputs=32)

    with tf.name_scope("conv1_visual_out"):
        conv1_out = tf.reshape(tensor=conv1, shape=[1, 28, 28, 32])
        conv1_out = tf.transpose(conv1_out, perm=[1, 2, 0, 3])
        conv1_out = tf.unstack(value=conv1_out, axis=3)
        tf.summary.image(name="conv1_out", tensor=conv1_out, max_outputs=32)


    with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            strides=1,
            use_bias=True,
            padding="same",
            activation=tf.nn.relu,
            kernel_size=[5, 5],
            name="conv2")
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.name_scope('dense'):
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense")
        dropout = tf.layers.dropout(inputs=dense, rate=0.45, training=True)

    with tf.name_scope('output'):
        y = tf.layers.dense(inputs=dropout, name="output", units=10)
        softmax_y = tf.nn.softmax(dropout, name="softmax_y")
        tf.summary.histogram(name="output_summary", values=y)

    with tf.name_scope('cross_entropy_distance'):
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=_y, logits=y)
        with tf.name_scope("total"):
            cost = tf.reduce_mean(cross_entropy)

    tf.summary.scalar("cross_entropy", cost)

    with tf.name_scope("train"):
        #minimize function
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_step = optimizer.minimize(cost, global_step=global_step)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', sess.graph)

    tf.global_variables_initializer().run()

    def generate_feed():
        return {x: x_batch, _y: y_batch}

    for i in range(10000):
        run_metadata = tf.RunMetadata()
        x_batch, y_batch = mnist.train.next_batch(1)
        summary, acc, _ = sess.run([merged, accuracy, train_step], feed_dict=generate_feed())
        if i % 100 == 0:
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))

    train_writer.close()

def main(_):
    train()


if __name__ == "__main__":
    tf.app.run(main=main)




