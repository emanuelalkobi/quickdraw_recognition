

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2

import argparse
import json
import os
import random
import sys
import numpy as np
import tensorflow as tf
import numpy as np
#print(path+'/fisr.png')

quick_draw = {0: 'axe', 1: 'cat', 2:'apple'}
batch_size = 128
num_of_classes=3
image_size=28
validate_data=3000

# The following defines a simple CovNet Model.
def SVHN_net_v0(x_):
    with tf.variable_scope("CNN"):
        conv1 = tf.layers.conv2d(
                                 inputs=x_,
                                 filters=32,  # number of filters
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)
            
        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2, 2], strides=2)  # convolution stride
        conv2 = tf.layers.conv2d(
                                  inputs=pool1,
                                  filters=32, # number of filters
                                  kernel_size=[5, 5],
                                  padding="same",
                                  activation=tf.nn.relu)
                                 
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=2)  # convolution stride
                                 
                                 
        pool_flat = tf.contrib.layers.flatten(pool2, scope='pool2flat')
        dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=num_of_classes)
        return tf.nn.softmax(logits)




x_ = tf.placeholder(tf.float32, [None, image_size, image_size,1],name='x')
y_=SVHN_net_v0(x_)

saver = tf.train.Saver()
print("variables are1",tf.trainable_variables())
with tf.Session() as sess:
    saver.restore(sess, "./saved_sess/model.ckpt")
    
    
    axe_data=np.load('axe.npy')
    test_im=axe_data[1:100]
    test_im=np.reshape(test_im,[test_im.shape[0],image_size,image_size,1])
    print(test_im.shape)
    res=sess.run(y_, feed_dict={x_:test_im})
    print(res.shape,res)
    print("probability result:[axe,cat,apple]")
    for i,prob in enumerate(res):
        print("test number ",i)
        for j in range(3):
            print("probability to be",quick_draw[j], ":%.16f" % prob[j])
        print("predictes result for test number",i,"is:",quick_draw[np.argmax(res[i])])
