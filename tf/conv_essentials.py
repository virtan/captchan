import tensorflow as tf
import numpy as np
import gc
from PIL import Image
import pprint
import os


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

def max_pool_custom(x, K, S):
    return tf.nn.max_pool(x, ksize=K,
            strides=S, padding='SAME')

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

