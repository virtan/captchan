# Convention:
#    x -- input tensor
#    y -- output tensor
#    y_expected -- learning output tensor
#    keep_prob -- dropout variable

import tensorflow as tf
import conv_essentials as ce

x = tf.placeholder(tf.float32, shape = [None, 60*60])
y_expected = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1,60,60,1])

w_conv1 = ce.weight_variable([5, 5, 1, 6])
b_conv1 = ce.bias_variable([6])
h_conv1 = tf.nn.relu(ce.conv2d(x_image, w_conv1, 'SAME') + b_conv1) # 60x60x1 -> 60x60x6
h_pool1 = ce.max_pool_2x2(h_conv1) # 60x60x6 -> 30x30x6

w_conv2 = ce.weight_variable([5, 5, 6, 16])
b_conv2 = ce.bias_variable([16])
h_conv2 = tf.nn.relu(ce.conv2d(h_pool1, w_conv2, 'VALID') + b_conv2) # 30x30x6 -> 26x26x16
h_pool2 = ce.max_pool_2x2(h_conv2) # 26x26x16 -> 13x13x16

w_conv3 = ce.weight_variable([13, 13, 16, 120])
b_conv3 = ce.bias_variable([120])
h_conv3 = tf.nn.relu(ce.conv2d(h_pool2, w_conv3, 'VALID') + b_conv3) # 13x13x16 -> 1x1200

h_pool2_flat = tf.reshape(h_conv3, [-1, 1*120])

w_fc1 = ce.weight_variable([1*120, 84])
b_fc1 = ce.bias_variable([84])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1) # 5x5x16 -> 1x120

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = ce.weight_variable([84, 10])
b_fc2 = ce.bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2) # 1x84 -> 1x10

