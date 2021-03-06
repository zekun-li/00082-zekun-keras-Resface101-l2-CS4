import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
os.environ['CUDA_VISIBLE_DEVICES'] = '4' 
from debug_face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.optimizers import SGD
from resnet101 import ResNet101


######### params #############
mean_img_file = 'model/keras_mean_img.npy' 
nb_classes = 62955

#####################################


#model = ResNet101(include_top=True, l2_norm = True, scale_param = 100, weights=None, classes = nb_classes) #weights=None for random initialization
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy']) ############# PARAMS NOT FINALIZED#####
#train_datagen = FaceAugDataGen(mode = 'training', batch_size=16,im_shape = (224,224), n_classes = nb_classes, source = '/lfs2/tmp/anh-train/', mean_file = '/lfs2/tmp/iacopo-train/average_face.bin' )
val_datagen = FaceAugDataGen(mode = 'validation', batch_size=16,im_shape = (224,224), n_classes = nb_classes, source = '/lfs2/tmp/anh-train/', mean_file = mean_img_file )
#model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, validation_data = val_datagen, validation_steps = 1000)
#print val_datagen[0]
import numpy as np
import cv2
mean_img = np.load('model/keras_mean_img.npy')
for j in range(2):
    for i in range(val_datagen[j][0].shape[0]): cv2.imwrite('new_keepratio_'+str(j) + '_'+str(i)+'.jpg',(val_datagen[j][0][i] + mean_img).transpose(1,2,0))
print 'done'
