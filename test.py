

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

image_size=28



# The following defines a simple CovNet Model.
def SVHN_net_v0(x_,num_of_classes):
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


def create_dic():
    dir_data='data/'
    dict={}
    i=0
    for file in sorted(os.listdir(dir_data)):
        if file.endswith(".npy"):
            str=file.split(".")
            print(str)
            dict[str[0]]=i
            i=i+1


    return i,dict

def test_cnn(cnn,test_img):
    print("train cnn started")
    x_ = tf.placeholder(tf.float32, [None, cnn.image_size, cnn.image_size,1],name='x')
    y_=SVHN_net_v0(x_,cnn.num_of_classes)
    y_pred=np.zeros(test_img.shape[0])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./saved_sess/model.ckpt")
        res=sess.run(y_, feed_dict={x_:test_img})
        print(res.shape)
        print("probability result:")
        for i,prob in enumerate(res):
            print("test number ",i)
            for j in range(cnn.num_of_classes):
                print("probability to be",cnn.dict[j], ":%.16f" % prob[j])
            print("predictes result for test number",i,"is:",cnn.dict[np.argmax(res[i])])
            y_pred[i]=(np.argmax(res[i]))
        return(y_pred)

#axe_data=np.load('axe.npy')
#test_im=axe_data[1:100]
#test_im=np.reshape(test_im,[test_im.shape[0],image_size,image_size,1])
#print(test_im.shape)
#train_cnn(test_im)
