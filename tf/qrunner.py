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
    pipeline = ce.input_pipeline(filenames, 50, 4, 300)

    y_reshaped = tf.reshape(m.y, [-1, 6, 10])
    y_lreshaped = tf.reshape(m.y, [-1, 10])
    y_expected_reshaped = tf.reshape(m.y_expected, [-1, 6, 10])
    y_expected_lreshaped = tf.reshape(m.y_expected, [-1, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_lreshaped, y_expected_lreshaped))
    tf.scalar_summary('cross_entropy', cross_entropy)
    #train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.reduce_all(tf.equal(tf.argmax(y_reshaped, 1), tf.argmax(y_expected_reshaped,1)), [1])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('summary/train', sess.graph)
    test_writer = tf.train.SummaryWriter('summary/test')

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess.run(init_op)
    threads = tf.train.start_queue_runners(coord = coord)
    saver = tf.train.Saver()

    try:
        i = 0
        while not coord.should_stop():
            batch_x, batch_y_expected = sess.run(pipeline)
            #if i%1000 == 0:
            #    print("step %d"%(i))
            #    print "first value: "
            #    print tf.reshape(batch_y_expected[0], [6, 10]).eval()
            #    ce.show_img(np.reshape(batch_x[0], (220,60)), 'pic')
            #    print "pic generated"
            if i%100 == 0:
                y_reshaped_, y_lreshaped_, y_expected_reshaped_, y_expected_lreshaped_, cross_entropy_, correct_prediction_, accuracy_, test_summary, test_accuracy = sess.run([y_reshaped, y_lreshaped, y_expected_reshaped, y_expected_lreshaped, cross_entropy, correct_prediction, accuracy, merged, accuracy], feed_dict = {
                        m.x: batch_x, m.y_expected: batch_y_expected,
                        m.keep_prob: 1.0})
                test_writer.add_summary(test_summary, i)
                print("step %d, training accuracy %g"%(i, test_accuracy))
                print "y_reshaped:\n"
                print y_reshaped
                print y_reshaped_
                print "y_lreshaped:\n"
                print y_lreshaped
                print y_lreshaped_
                print "y_expected_reshaped:\n"
                print y_expected_reshaped
                print y_expected_reshaped_
                print "y_expected_lreshaped:\n"
                print y_expected_lreshaped
                print y_expected_lreshaped_
                print "cross_entropy:\n"
                print cross_entropy
                print cross_entropy_
                print "correct_prediction:\n"
                print correct_prediction
                print correct_prediction_
                print "accuracy:\n"
                print accuracy
                print accuracy_
                save_path = saver.save(sess, "model.ckpt")
            train_summary, _ = sess.run([merged, train_op], feed_dict = {
                m.x: batch_x, m.y_expected: batch_y_expected, 
                m.keep_prob: 0.5})
            train_writer.add_summary(train_summary, i)
            i += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        save_path = saver.save(sess, "model.ckpt")
        print("Model saved in file: %s" % save_path)
        sess.close()
