#!/usr/bin/env python

import tensorflow as tf
import os.path

def read_pngs(filename_queue):
    image_reader = tf.WholeFileReader()
    image_filename, image_file = image_reader.read(filename_queue)
    print(image_filename.eval())
    image = tf.image.decode_png(image_file, channels=1)
    image = tf.reshape(image, [220, 60, 1])
    label_reversed_str = tf.reverse(tf.decode_raw(image_filename, tf.uint8),
            [True])[4:10]
    label_number = tf.reverse(tf.sub(tf.to_int32(label_reversed_str),
            ord('0')), [True])
    #print(label_number.eval())
    label = tf.one_hot(label_number, 10, 1.0, 0.0, -1)
    label = tf.reshape(label, [60])
    print(label.eval())
    #print(image.eval())
    return image, label

def input_pipeline(filenames, batch_size, read_threads, num_epochs = None):
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs = num_epochs, shuffle = True)
    return read_pngs(filename_queue)
    #image_list = [read_pngs(filename_queue)
                  #for _ in range(read_threads)]
    return 0
    #print(image_list)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    # min_after_dequeue = 10
    # capacity = min_after_dequeue + 3 * batch_size
    # image_batch, label_batch = tf.train.shuffle_batch_join(
    #   image_list, batch_size = batch_size, capacity = capacity,
    #   min_after_dequeue = min_after_dequeue)
    # return image_batch, label_batch


filenames = tf.train.match_filenames_once("../images/i_*.png")
sess = tf.Session()

with sess.as_default():
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    pipeline = input_pipeline(filenames, 50, 1, 1)
    #print(pipeline[1].eval())
    #image_tensor = sess.run(pipeline[0])
    #print(image_tensor)

##     num_preprocess_threads = 1
##     min_queue_examples = 256
##     images = tf.train.shuffle_batch(
##         [image],
##         batch_size=batch_size,
##         num_threads=num_preprocess_threads,
##         capacity=min_queue_examples + 3 * batch_size,
##         min_after_dequeue=min_queue_examples)
##     image_tensor = sess.run(images)
##     print(image_tensor)
## 
##     coord.request_stop()
##     coord.join(threads)

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
