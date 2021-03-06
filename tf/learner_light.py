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

def branch(x):
    b_drop = tf.nn.dropout(x, keep_prob)
    b_drop_r = tf.reshape(b_drop, [-1, 6*256])
    W_fc1 = weight_variable([6 * 256, 256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.relu(tf.matmul(b_drop_r, W_fc1) + b_fc1)
    h_fc1r = tf.reshape(h_fc1, [-1, 256])
    return h_fc1r

x = tf.placeholder(tf.float32, shape=[None, 60 * 60])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,60,60,1]) # 60x60x1

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 60x60x1 -> 60x60x32
h_pool1 = max_pool_2x2(h_conv1) # 60x60x32 -> 30x30x32

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 30x30x32 -> 30x30x64
h_pool2 = max_pool_2x2(h_conv2) # 30x30x64 -> 15x15x64

h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])

W_id = weight_variable([15 * 15 * 64, 1024])
b_id = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_id) + b_id)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# net = tf.reshape(pre_net, [-1, 1])
# net6 = tf.tile(net, [1, 6])
# b0,b1,b2,b3,b4,b5 = tf.split(1, 6, net6)
# 
# branches = []
# branches.append(branch(b0))
# branches.append(branch(b1))
# branches.append(branch(b2))
# branches.append(branch(b3))
# branches.append(branch(b4))
# branches.append(branch(b5))
# for i in range(6):
#     b_drop = tf.nn.dropout(net, keep_prob)
#     W_fc1 = weight_variable([6 * 256, 256])
#     b_fc1 = bias_variable([256])
#     h_fc1 = tf.nn.relu(tf.matmul(b_drop, W_fc1) + b_fc1)
#     h_fc1r = tf.reshape(h_fc1, [-1, 256])
#     #W_fc2 = weight_variable([1024, 1])
#     #b_fc2 = bias_variable([1])
#     #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#     branches.append(h_fc1r)

#y_ = tf.pack(branches, axis=0)
#concatenated = tf.concat(0, branches)
# conc_flat = tf.reshape(net, [-1, 256])
# conc_drop = tf.nn.dropout(conc_flat, keep_prob)
# W_final = weight_variable([256, 1])
# b_final = bias_variable([1])
# y = tf.nn.softmax(tf.matmul(conc_drop, W_final) + b_final)
# y = net

pprint.pprint(y)

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# reduced_difference = tf.reduce_mean(tf.abs(tf.sub(y_, y)))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(reduced_difference)
# correct_prediction = tf.equal(tf.round(y_), tf.round(y))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print "Loading values..."
offset = 0
raw_values = np.loadtxt("../c10k_values.gz")
amount = raw_values.size / 1
print amount, "loaded"
print "Reshaping..."
values = np.reshape(raw_values, (amount, 1))
gc.collect()
print "Loading data..."
raw_data = np.loadtxt("../c10k_data.gz")
print "Reshaping..."
data = np.reshape(raw_data, (amount, 60*60))
raw_data = 0
print "Loaded"

gc.collect()
print "Garbage collected"

print "Medium pic:"
print data[amount/2,:]
print "Medium values:"
print values[amount/2,:]

print "Starting training"

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

for i in range(100*amount/50):
    batch = next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: data, y_: values, keep_prob: 1.0}))
