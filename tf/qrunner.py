#!/usr/bin/env python

import tensorflow as tf

def read_pngs(filename_queue):
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_png(image_file, channels=1)
    label_str = tf.decode_raw(image_file, tf.uint8)
    label_number = tf.sub(tf.to_int32(label_str[2:8]), ord('0'))
    label = tf.one_hot(label_number, 10, 1.0, 0.0, -1)
    return image, label

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    image_list = [read_pngs(filename_queue)
                  for _ in range(read_threads)]
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(
      image_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("../images/i_*.png"))
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)

# image = tf.image.decode_jpeg(image_file)
image = tf.image.decode_png(image_file)
image.set_shape((220, 60, 3))

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_preprocess_threads = 1
    min_queue_examples = 256
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    image_tensor = sess.run(images)
    print(image_tensor)

    coord.request_stop()
    coord.join(threads)

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
