#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gc
from PIL import Image
import pprint

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, P):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=P)
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='VALID')
            #strides=[1, 2, 2, 1], padding='SAME')

def show_img(data, name):
    w, h = data.shape
    data2 = np.empty((h,w,3), dtype = np.uint8)
    data2[:, :, 2] = data2[:, :, 1] = data2[:, :, 0] = data * 255
    img = Image.fromarray(data2, 'RGB')
    img.save(name + '.png')
    img.show()

saver = tf.train.Saver()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 60*60])
#x = tf.placeholder(tf.float32, shape=[None, 32*32])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])

#x_image = tf.reshape(x, [-1,60,60,1])
x_image = tf.reshape(x, [-1,60,60,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'SAME') + b_conv1) # 60x60x1 -> 60x60x6
h_pool1 = max_pool_2x2(h_conv1) # 60x60x6 -> 30x30x6

W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'VALID') + b_conv2) # 30x30x6 -> 26x26x16
h_pool2 = max_pool_2x2(h_conv2) # 26x26x16 -> 13x13x16

W_conv3 = weight_variable([13, 13, 16, 120])
b_conv3 = bias_variable([120])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 'VALID') + b_conv3) # 13x13x16 -> 1x1200

W_fc1 = weight_variable([1*120, 84])
#W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([84])

h_pool2_flat = tf.reshape(h_conv3, [-1, 1*120])
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 5x5x16 -> 1x120

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([84, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 1x84 -> 1x10

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

print "One-hotting values..."
values = np.reshape(values, (amount))
values2 = np.zeros((amount, 10))
values2[np.arange(amount), values.astype(int)] = 1.0
values = np.reshape(values2, (amount, 10))
values2 = 0

gc.collect()
print "Garbage collected"

mp = amount/2
print "Medium pic:"
print data[mp,:]
print "Medium values:"
print values[mp,:]
show_img(np.reshape(data[mp,:], (60,60)), 'medium_pic_1')
mp += 1
print "Medium+1 pic:"
print data[mp,:]
print "Medium+1 values:"
print values[mp,:]
show_img(np.reshape(data[mp,:], (60,60)), 'medium_pic_2')
mp += 1
print "Medium+2 pic:"
print data[mp,:]
print "Medium+2 values:"
print values[mp,:]
show_img(np.reshape(data[mp,:], (60,60)), 'medium_pic_3')

print "Starting training"

epochs_completed = 0

def next_batch(batch_size):
    global offset, data, values, epochs_completed
    if offset + batch_size >= amount:
        epochs_completed += 1
        print("epoch %d completed"%epochs_completed)
        if epochs_completed == 3:
            return False
        perm = np.arange(amount)
        np.random.shuffle(perm)
        values = values[perm]
        data = data[perm]
        offset = 0
    start = offset
    offset += batch_size
    end = offset
    return data[start:end], values[start:end]

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# reduced_difference = tf.reduce_mean(tf.abs(tf.sub(y_, y_conv)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(2000000):
    batch = next_batch(50)
    if batch == False:
        break
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: data, y_: values, keep_prob: 1.0}))

save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables())
# for i in range(20000):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 
# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
