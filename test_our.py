

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
import test as test

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



def test_cnn(test_img):
    print("train cnn started")
    x_ = tf.placeholder(tf.float32, [None, image_size, image_size,1],name='x')
    y_=SVHN_net_v0(x_)
    y_pred=np.zeros(test_img.shape[0])
    saver = tf.train.Saver()
    print("variables are1",tf.trainable_variables())
    with tf.Session() as sess:
        saver.restore(sess, "./saved_sess/model.ckpt")
        res=sess.run(y_, feed_dict={x_:test_img})
        print(res.shape,res)
        print("probability result:[axe,cat,apple]")
        for i,prob in enumerate(res):
            print("test number ",i)
            for j in range(3):
                print("probability to be",quick_draw[j], ":%.16f" % prob[j])
            print("predictes result for test number",i,"is:",quick_draw[np.argmax(res[i])])
            y_pred[i]=(np.argmax(res[i]))
        return(y_pred)

#axe_data=np.load('axe.npy')
#test_im=axe_data[1:100]
#test_im=np.reshape(test_im,[test_im.shape[0],image_size,image_size,1])
#print(test_im.shape)
#train_cnn(test_im)
#cwd = os.getcwd()
quick_draw = {0: 'axe', 1: 'cat', 2:'apple'}
reversed_quik_draw = dict(map(reversed, quick_draw.items()))

files_num=0
for i,file in enumerate(os.listdir('test_img/')):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") :
        files_num=files_num+1

#batch=np.zeros((files_num,image_size,image_size))
#print(batch.shape)
#labels=np.zeros(files_num)

#for i,file in enumerate(os.listdir('test_img/')):
#    filename = os.fsdecode(file)
#    if filename.endswith(".jpg") :
#        print("file name!!!!!!!!!!!!!!!!!!!!!!!",filename,i)
#        img_name=filename
#        img = cv2.imread('test_img/'+img_name,0)
#        img=255-img
#        print(img.shape)
#        resized_image = cv2.resize(img, (image_size, image_size),interpolation = cv2.INTER_CUBIC)
#        cv2.imwrite('process_img/'+img_name, resized_image)
        #resized_image=np.expand_dims(resized_image,0)
        #resized_image=np.expand_dims(resized_image,3)
#        print(resized_image.shape,"-0-0--0-")


#print(batch.shape,"000000000")
#batch=np.expand_dims(batch,3)
#print(batch.shape,"000000000")

#y_predicted=test.test_cnn(batch)


img_name='axe2.jpg'
img = cv2.imread('test_img/'+img_name,0)
img=255-img
print(img.shape)
resized_image = cv2.resize(img, (image_size, image_size),interpolation = cv2.INTER_CUBIC)
cv2.imwrite('process_img/'+img_name+'.jpg', resized_image)
resized_image=np.expand_dims(resized_image,0)
resized_image=np.expand_dims(resized_image,3)
y_predicted=test.test_cnn(resized_image)
