

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
        self.dir_data='../data/'
        self.num_of_classes,self.dict =create_dic(self.dir_data)
        self.image_size = 28
        self.validate_data = 3000



#axe_data=np.load('axe.npy')
#test_im=axe_data[1:100]
#test_im=np.reshape(test_im,[test_im.shape[0],image_size,image_size,1])
#print(test_im.shape)
#train_cnn(test_im)
#cwd = os.getcwd()
#quick_draw = {0: 'axe', 1: 'cat', 2:'apple',3:'butterfly',4:'carrot',5:'clock'}
#reversed_quik_draw = dict(map(reversed, quick_draw.items()))

#files_num=0
#for i,file in enumerate(os.listdir('test_img/')):
#    filename = os.fsdecode(file)
#    if filename.endswith(".jpg") :
#        files_num=files_num+1

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
def main(args):
    quick_draw_cnn=cnn()
    img = cv2.imread(args.draw_path,0)
    print(img.shape)
    num_of_classess,dict=create_dic('../data/')
    resized_image = cv2.resize(img, (quick_draw_cnn.image_size, quick_draw_cnn.image_size),interpolation = cv2.INTER_CUBIC)
    resized_image=np.expand_dims(resized_image,0)
    resized_image=np.expand_dims(resized_image,3)
    y_predicted=test.test_cnn(quick_draw_cnn,resized_image,1)
    print(dict)
    print("You inserted an ",dict[int(args.label)],"draw and our CNN predicted it as ",dict[int(y_predicted)-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--draw_path', )
    parser.add_argument('-lab', '--label', )
    args = parser.parse_args()
    print("-----------------------------------------------------------------------")
    print("Try to identify draw: " ,args.draw_path)
    print("-----------------------------------------------------------------------")
    main(args)

