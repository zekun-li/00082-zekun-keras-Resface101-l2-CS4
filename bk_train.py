import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.optimizers import SGD
from resnet101 import ResNet101
from keras.callbacks import ModelCheckpoint,CSVLogger
import tensorflow as tf

######### params #############
mean_img_file = 'model/keras_mean_img.npy' 
nb_classes = 62955
saved_weights = 'weights/1/10-9.52.hdf5'
batch_size = 32
#####################################


model = ResNet101(include_top=True, l2_norm = True, scale_param = 100, weights=None, classes = nb_classes) #weights=None for random initialization
sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy']) ############# PARAMS NOT FINALIZED#####


# load weights
if saved_weights is not None:
    assert os.path.isfile(saved_weights)
    model.load_weights(saved_weights)

# generators
train_datagen = FaceAugDataGen(mode = 'training', batch_size=batch_size,im_shape = (224,224), n_classes = nb_classes, source = '/lfs2/tmp/anh-train/', mean_file = mean_img_file )
val_datagen = FaceAugDataGen(mode = 'validation', batch_size=batch_size,im_shape = (224,224), n_classes = nb_classes, source = '/lfs2/tmp/anh-train/', mean_file = mean_img_file )

# callbacks
csv_logger = CSVLogger('logs/train.log')     
model_save_path = 'weights/{epoch:02d}-{val_loss:.2f}.hdf5' 
model_save_best = 'weights/best-{epoch:02d}-{val_loss:.2f}.hdf5'  
check_point = ModelCheckpoint(model_save_path, save_best_only=False,save_weights_only=False,period = 10)   
check_point_best = ModelCheckpoint(model_save_best, save_best_only=True,save_weights_only=False)


model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = val_datagen, validation_steps = 1000, callbacks =[csv_logger, check_point, check_point_best])
