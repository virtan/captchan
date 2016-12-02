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


def put_kernels_on_grid(kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8) 
