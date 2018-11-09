

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

import test as test
#print(path+'/fisr.png')


batch_size = 128
num_of_classes=6
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
        return logits


def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, image_size, image_size,1],name='x')
            y_ = tf.placeholder(tf.int32, [None],name='y')
            y_logits = model_function(x_)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            trainer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = trainer.minimize(cross_entropy_loss)
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                               'accuracy': accuracy, 'loss': cross_entropy_loss}

    return model_dict


def get_data(x,y,i):
    start=i*batch_size
    end=start+batch_size
    if (end>x.shape[0]):
        end=x.shape[0]
            #print("start is ",start)
            #print("end is ",end)
    x_batch_data=x[start:end,:,:,:]
    y_batch_data=y[start:end]
    return x_batch_data,y_batch_data



def train_model(model_dict, x_data,y_data,x_test,y_test ,epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        saver = tf.train.Saver()
        #print("variables are",tf.trainable_variables())
        sess.run(tf.global_variables_initializer())
        batch_num=int(np.ceil(x_data.shape[0]/batch_size))
        for epoch_i in range(epoch_n):
            for iter_i in range(batch_num):
                x_placeholder=model_dict['inputs'][0]
                y_placeholder=model_dict['inputs'][1]
                #train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                [x_batch_data,y_batch_data]=get_data(x_data,y_data,iter_i)
                sess.run(model_dict['train_op'], feed_dict={x_placeholder:x_batch_data,y_placeholder:y_batch_data})

                if (iter_i%200==0):
                    to_compute = [model_dict['loss'], model_dict['accuracy']]
                    loss,accuracy=sess.run(to_compute, feed_dict={x_placeholder:x_test,y_placeholder:y_test})
                    print(iter_i,"/",batch_num,"loss:",loss," accuracy:",accuracy)
        saver.save(sess, "./saved_sess/model.ckpt")

def load_data():
    dir_data='data/'
    axe_data=np.load(dir_data+'axe.npy')
    cat_data=np.load(dir_data+'cat.npy')
    apple_data=np.load(dir_data+'apple.npy')
    butterfly_data=np.load(dir_data+'butterfly.npy')
    carrot_data=np.load(dir_data+'carrot.npy')
    clock_data=np.load(dir_data+'clock.npy')


    #labels are 0-axe 1-cat 2-apple
    axe_labels=np.zeros(axe_data.shape[0])*0
    cat_labels=np.ones(cat_data.shape[0])*1
    apple_labels=np.ones(apple_data.shape[0])*2
    butterfly_labels=np.zeros(butterfly_data.shape[0])*3
    carrot_labels=np.ones(carrot_data.shape[0])*4
    clock_labels=np.ones(clock_data.shape[0])*5
    
    #connect all data for randomization

    data_d=np.concatenate((axe_data,cat_data,apple_data,butterfly_data,carrot_data,clock_data))
    data_l=np.concatenate((axe_labels,cat_labels,apple_labels,butterfly_labels,carrot_labels,clock_labels))
    data_l=np.expand_dims(data_l,1)
    data_all=np.concatenate((data_d,data_l),axis=1)
    data_all=np.random.permutation(data_all)

    x_data=data_all[:,0:-1]
    y_data=data_all[:,-1]
    num_img=x_data.shape[0]
    data_img=np.reshape(x_data,[num_img,image_size,image_size])
   
    
    data_train=data_img[validate_data:,:,:]
    data_train=np.expand_dims(data_train,3)

    labels_train=y_data[validate_data:]
    data_test=data_img[:validate_data:,:,:]
    data_test=np.expand_dims(data_test,3)

    labels_test=y_data[:validate_data]

    
    return data_train,labels_train,data_test,labels_test

[x_data,y_data,x_test,y_test]=load_data()

print("----------_#$%------")
print(x_data.shape)
print(y_data.shape)
print(x_test.shape)
print(y_test.shape)
model_dict = apply_classification_loss(SVHN_net_v0)
train_model(model_dict, x_data,y_data,x_test,y_test ,epoch_n=1, print_every=20)

#test test data after finishing training 
y_predicted=test.test_cnn(x_test)
print(x_test.shape)
print("predicted is :",(y_predicted.shape),y_test.shape)
mistakes=np.nonzero(y_predicted-y_test)
#mistakes is tuple,take the array only
mistakes=mistakes[0]
#print(mistakes[0],type(mistakes[0]))
error_rate=mistakes.shape[0]/y_test.shape[0]
print("accuracy is :",1-error_rate)
