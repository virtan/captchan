import tensorflow as tf
import numpy as np
from PIL import Image


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# P could be SAME (keep size the same) or VALID
def conv2d(x, W, P):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=P)

def conv2d_strides(x, W, P, S):
    return tf.nn.conv2d(x, W, strides=S, padding=P)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

def show_img(data, name):
    w, h = data.shape
    data2 = np.empty((w,h,3), dtype = np.uint8)
    data2[:, :, 2] = data2[:, :, 1] = data2[:, :, 0] = data * 255
    #img = Image.fromarray(np.transpose(data2, (1,0,2)), 'RGB')
    img = Image.fromarray(np.reshape(data2, (h,w,3)), 'RGB')
    #img.save(name + '.png')
    img.show()

def mediums(values, data, amount):
    mp = amount/2
    print "Medium pic:"
    print data[mp,:]
    print "Medium values:"
    print values[mp,:]
    #show_img(np.reshape(data[mp,:], (60,60)), 'medium_pic_1')
    mp += 1
    print "Medium+1 pic:"
    print data[mp,:]
    print "Medium+1 values:"
    print values[mp,:]
    #show_img(np.reshape(data[mp,:], (60,60)), 'medium_pic_2')
    mp += 1
    print "Medium+2 pic:"
    print data[mp,:]
    print "Medium+2 values:"
    print values[mp,:]
    #show_img(np.reshape(data[mp,:], (60,60)), 'medium_pic_3')

def read_pngs(filename_queue):
    image_reader = tf.WholeFileReader()
    image_filename, image_file = image_reader.read(filename_queue)
    #print(image_filename.eval())
    image = tf.image.decode_png(image_file, channels=1)
    image = tf.reshape(image, [220, 60, 1])
    image = tf.div(tf.to_float(image), 255)
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
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(
      image_list, batch_size = batch_size, capacity = capacity,
      min_after_dequeue = min_after_dequeue)
    return image_batch, label_batch

