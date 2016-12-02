#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from PIL import Image

import conv_essentials as ce
import model_220x60 as m

sess = tf.Session()

with sess.as_default():
    coord = tf.train.Coordinator()
    filenames = tf.train.match_filenames_once("../images/i_*.png")
    pipeline = ce.input_pipeline(filenames, 1, 28, 1)
    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess.run(init_op)
    threads = tf.train.start_queue_runners(coord = coord)
    saver.restore(sess, "model.ckpt")

    grid_x = 12
    grid_y = 10
    grid = ce.put_kernels_on_grid(m.w_conv3, grid_y, grid_x)
    grid_flat = tf.reshape(tf.slice(grid, [0, 0, 0, 10], [1, 150, 180, 1]), [180, 150])
    ce.show_img(grid_flat.eval(), "h_conv_2/features")

    while not coord.should_stop():
        raw_input('Enter to test next random pic...')
        print
        batch_x, batch_y_expected = sess.run(pipeline)
        print sess.run(tf.argmax(tf.reshape(batch_y_expected[0], (6, 10)), 1))
        print tf.reshape(batch_y_expected[0], [6, 10]).eval()
        print sess.run(tf.argmax(tf.reshape(m.y, (6, 10)), 1), feed_dict={m.x: batch_x, m.keep_prob: 1})
        print sess.run(tf.round(tf.nn.top_k(tf.reshape(m.y, (6, 10)), 10, False).values * 100), feed_dict={m.x: batch_x, m.keep_prob: 1})
        #ce.show_img(np.reshape(batch_x[0], (220,60)), 'pic')
        print "pic generated\n"

