#!/usr/bin/env python

import tensorflow as tf
import numpy as np

x = tf.constant("i_123456.png")
#x = tf.reshape(x, (-1, 1))
x = tf.decode_raw(x, tf.uint8)
y = tf.sub(tf.to_int32(x[2:8]), ord('0'))
yh = tf.one_hot(y, 10, 1.0, 0.0, -1)

#z = tf.zeros((10))
#z[tf.arange(6), values.astype(int)] = 1.0
#values = np.reshape(values2, (amount, 6*10))
#values2 = 0

sess = tf.InteractiveSession()
print(yh.eval())
