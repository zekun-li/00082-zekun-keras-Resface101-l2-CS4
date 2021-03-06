import numpy as np 
import warnings    
import os          
import cv2
os.environ['KERAS_BACKEND'] = 'tensorflow' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   
import keras
import sys
sys.path.insert(0,'/nfs/isicvlnas01/projects/glaive/expts/00082-zekun-keras-Resface101-l2-CS4/')
import resnet101

model = resnet101.ResNet101(include_top=False, l2_norm = True, scale_param = 100, weights='imagenet')
model.load_weights('../model/keras_model.hdf5')
mean_img = np.load('../model/keras_mean_img.npy')

img = cv2.imread('10000_frames-125486.jpg')
img = cv2.resize(img, (224,224))
# BGR order, [0,225] range
# from channels_last to channels_first
img = img.transpose(2,0,1)
img = img.astype('float32')
img = img - mean_img
#img = img/255.



print model.predict(np.expand_dims(img, axis = 0))
