

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



def create_dic(dir_data):
    dict={}
    i=0
    for file in sorted(os.listdir(dir_data)):
        if file.endswith(".npy"):
            str=file.split(".")
            dict[i]=str[0]
            i=i+1
    return i,dict

class cnn:
    def __init__(self):
        self.batch_size = 128
        self.dir_data='data/'
        self.num_of_classes,self.dict =create_dic(self.dir_data)
        self.image_size = 28
        self.validate_data = 3000



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
        return logits


def apply_classification_loss(cnn,model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, cnn.image_size, cnn.image_size,1],name='x')
            y_ = tf.placeholder(tf.int32, [None],name='y')
            y_logits = model_function(x_,cnn.num_of_classes)
            
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


def get_data(cnn,x,y,i):
    start=i*cnn.batch_size
    end=start+cnn.batch_size
    if (end>x.shape[0]):
        end=x.shape[0]
    x_batch_data=x[start:end,:,:,:]
    y_batch_data=y[start:end]
    return x_batch_data,y_batch_data



def train_model(cnn,model_dict, x_data,y_data,x_test,y_test ,epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        batch_num=int(np.ceil(x_data.shape[0]/cnn.batch_size))
        for epoch_i in range(epoch_n):
            for iter_i in range(batch_num):
                x_placeholder=model_dict['inputs'][0]
                y_placeholder=model_dict['inputs'][1]
                #train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                [x_batch_data,y_batch_data]=get_data(cnn,x_data,y_data,iter_i)
                sess.run(model_dict['train_op'], feed_dict={x_placeholder:x_batch_data,y_placeholder:y_batch_data})

                if (iter_i%200==0):
                    to_compute = [model_dict['loss'], model_dict['accuracy']]
                    loss,accuracy=sess.run(to_compute, feed_dict={x_placeholder:x_test,y_placeholder:y_test})
                    print(iter_i,"/",batch_num,"loss:",loss," accuracy:",accuracy)
        saver.save(sess, "./saved_sess/model.ckpt")



def load_data(cnn):
    dir_data='data/'
    num_of_classess,dict=create_dic(dir_data)
    data_l=np.zeros((1))
    data_d=np.zeros((1,cnn.image_size*cnn.image_size))
    index=0
    for file in sorted(os.listdir(dir_data)):
        if file.endswith(".npy"):
            print(data_l.shape,data_d.shape,"cur label num!",index)
            curr_data=np.load(dir_data+file)
            data_d=np.concatenate((data_d,curr_data))
            data_l=np.concatenate((data_l,np.ones(curr_data.shape[0])*index))
            index=index+1





    
    data_l=np.expand_dims(data_l,1)
    data_all=np.concatenate((data_d,data_l),axis=1)
    data_all=np.random.permutation(data_all)

    x_data=data_all[:,0:-1]
    y_data=data_all[:,-1]
    num_img=x_data.shape[0]
    data_img=np.reshape(x_data,[num_img,cnn.image_size,cnn.image_size])
   
    
    data_train=data_img[cnn.validate_data:,:,:]
    data_train=np.expand_dims(data_train,3)

    labels_train=y_data[cnn.validate_data:]
    data_test=data_img[:cnn.validate_data:,:,:]
    data_test=np.expand_dims(data_test,3)

    labels_test=y_data[:cnn.validate_data]

    
    return data_train,labels_train,data_test,labels_test


def main():
    quick_draw_cnn=cnn()
    [x_data,y_data,x_test,y_test]=load_data(quick_draw_cnn)

    print("----------_#$%------")
    print(x_data.shape)
    print(y_data.shape)
    print(x_test.shape)
    print(y_test.shape)
    model_dict = apply_classification_loss(quick_draw_cnn,SVHN_net_v0)
    train_model(quick_draw_cnn,model_dict, x_data,y_data,x_test,y_test ,epoch_n=1, print_every=20)

    #test test data after finishing training
    y_predicted=test.test_cnn(quick_draw_cnn,x_test)

    mistakes=np.nonzero(y_predicted-y_test)
    #mistakes is tuple,take the array only
    mistakes=mistakes[0]
    print(mistakes[0],type(mistakes[0]))
    error_rate=mistakes.shape[0]/y_test.shape[0]
    print("accuracy is :",1-error_rate)


if __name__ == '__main__':
    main()
