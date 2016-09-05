#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gc
from PIL import Image
import pprint

import conv_essentials as ce
sess = tf.Session()
import model_60x60 as m
import load_values as lv

saver = tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "model.ckpt")
    print("Model restored.")
    # Do some work with the model
    values, data, amount = lv.load_data_n_values('../c1k')
    while True:
        mode=int(raw_input('Input: '))
        print(values[mode,:])
        ce.show_img(np.reshape(data[mode,:], (60,60)), 'tmppic')
        print(sess.run(tf.round(tf.nn.top_k(m.y, 10, False).values * 100), feed_dict={m.x: np.reshape(data[mode,:], (-1, 60*60)), m.keep_prob: 1}))

