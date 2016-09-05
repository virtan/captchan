#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gc
from PIL import Image
import pprint
import os

import conv_essentials as ce

sess = tf.Session()

import model_60x60 as m

import load_values as lv

values, data, amount = lv.load_data_n_values('../c1k')

print "One-hotting values ..."
values = np.reshape(values, (amount))
values2 = np.zeros((amount, 10))
values2[np.arange(amount), values.astype(int)] = 1.0
values = np.reshape(values2, (amount, 10))
values2 = 0
gc.collect()

ce.mediums(values, data, amount)

print "Starting training"

offset = 0
epochs_completed = 0

def next_batch(batch_size):
    global offset, data, values, epochs_completed
    if offset + batch_size >= amount:
        epochs_completed += 1
        print("epoch %d completed"%epochs_completed)
        if epochs_completed == 80:
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

cross_entropy = tf.reduce_mean(-tf.reduce_sum(m.y_expected * tf.log(m.y), reduction_indices=[1]))
#ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
# reduced_difference = tf.reduce_mean(tf.abs(tf.sub(y_, y_conv)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(m.y,1), tf.argmax(m.y_expected,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy_summary = tf.scalar_summary("accuracy", accuracy)

#merged = tf.merge_all_summaries()
#writer = tf.train.SummaryWriter("summary", sess.graph)

#tf.initialize_all_variables().run()

init_op = tf.initialize_all_variables()
sess.run(init_op)
saver = tf.train.Saver()

# for i in range(1000):
#   if i % 10 == 0:  # Record summary data, and the accuracy
#     feed = {x: data, y_: values, keep_prob: 0.5}
#     result = sess.run([merged, accuracy], feed_dict=feed)
#     summary_str = result[0]
#     acc = result[1]
#     writer.add_summary(summary_str, i)
#     print("Accuracy at step %s: %s" % (i, acc))
#   else:
#     batch_xs, batch_ys = next_batch(100)
#     feed = {x: batch_xs, y_: batch_ys, keep_prob: 0.5}
#     sess.run(train_step, feed_dict=feed)

for i in range(2000000):
    batch = next_batch(50)
    if batch == False:
        break
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={
            m.x:batch[0], m.y_expected: batch[1], m.keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess,
            feed_dict={m.x: batch[0], m.y_expected: batch[1], m.keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    m.x: data, m.y_expected: values, m.keep_prob: 1.0}))

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
