#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gc
import pprint

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 220 * 60])
y_ = tf.placeholder(tf.float32, shape=[None, 6])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,220,60,1]) # 220x60x1

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 220x60x1 -> 220x60x32
h_pool1 = max_pool_2x2(h_conv1) # 220x60x32 -> 110x30x32

W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 110x30x32 -> 110x30x32
h_pool2 = max_pool_2x2(h_conv2) # 110x30x32 -> 55x15x32

h_pool2_flat = tf.reshape(h_pool2, [-1, 55*15*32])

branches = []
for i in range(0, 6):
    W_fc1 = weight_variable([55 * 15 * 32, 256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([256, 1])
    b_fc2 = bias_variable([1])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    branches.append(y_conv)

#y_ = tf.pack(branches, axis=0)
y_ = tf.concat(1, branches)
pprint.pprint(y_)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.round(y_conv), tf.round(y_))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

offset = 0
raw_values = np.loadtxt("../abc_values.gz")
amount = raw_values.size / 6
values = np.reshape(raw_values, (amount, 6))
gc.collect()
data = np.reshape(np.loadtxt("../abc_data.gz"), (amount, 220*60))
gc.collect()

epochs_completed = 0

def next_batch(batch_size):
    global offset, data, values, epochs_completed
    if offset + batch_size > amount:
        epochs_completed += 1
        perm = np.arange(amount)
        values = values[perm]
        data = data[perm]
        offset = 0
    start = offset
    offset += 50
    end = offset
    return data[start:end], values[start:end]

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: data, y_: values, keep_prob: 1.0}))
