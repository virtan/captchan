#!/usr/bin/env python

import tensorflow as tf
import conv_essentials as ce
import model_220x60 as m

def read_pngs(filename_queue):
    image_reader = tf.WholeFileReader()
    image_filename, image_file = image_reader.read(filename_queue)
    #print(image_filename.eval())
    image = tf.image.decode_png(image_file, channels=1)
    image = tf.reshape(image, [220, 60, 1])
    image = tf.to_float(image)
    label_reversed_str = tf.reverse(tf.decode_raw(image_filename, tf.uint8),
            [True])[4:10]
    label_number = tf.reverse(tf.sub(tf.to_int32(label_reversed_str),
            ord('0')), [True])
    #print(label_number.eval())
    label = tf.one_hot(label_number, 10, 1.0, 0.0, -1)
    label = tf.reshape(label, [60])
    #print(label.eval())
    #print(image.eval())
    return image, label

def input_pipeline(filenames, batch_size, read_threads, num_epochs = None):
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs = num_epochs, shuffle = True)
    image_list = [read_pngs(filename_queue)
                  for _ in range(read_threads)]
    #print(image_list)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(
      image_list, batch_size = batch_size, capacity = capacity,
      min_after_dequeue = min_after_dequeue)
    return image_batch, label_batch


sess = tf.Session()

with sess.as_default():
    coord = tf.train.Coordinator()
    filenames = tf.train.match_filenames_once("../images/i_*.png")
    pipeline = input_pipeline(filenames, 50, 4, 10)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(m.y_expected * tf.log(m.y + 1e-10), reduction_indices=[1]))
    #ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
    # reduced_difference = tf.reduce_mean(tf.abs(tf.sub(y_, y_conv)))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(tf.reshape(m.y, (-1, 6, 10)), 1), tf.argmax(tf.reshape(m.y_expected, (-1, 6, 10)),1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess.run(init_op)
    threads = tf.train.start_queue_runners(coord = coord)

    try:
        while not coord.should_stop():
            train_op.run(feed_dict = {
                m.x: pipeline[0].eval(), m.y_expected: pipeline[1].eval(),
                m.keep_prob: 1.0})

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()

# try:
#     while not coord.should_stop():
#         sess.run(train_op)
# 
# except tf.errors.OutOfRangeError:
#         print('Done training -- epoch limit reached')
# 
# finally:
#     coord.request_stop()
#     coord.join(threads)
#     sess.close()
