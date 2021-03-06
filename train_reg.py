import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 
from face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from resnet101_reg import ResNet101
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np

######### params #############
mean_img_file = 'model/keras_mean_img.npy' 
nb_classes = 62955
nb_gpus = 4
b_size = 128
saved_weights = 'weights/4/multi-best-12-3.29.hdf5'
#1/best-13-9.92.hdf5'
#3/multi-50-4.66.hdf5

csv_logger = CSVLogger('logs/reg_multi_train.log')     
model_save_path = 'weights/reg_multi-{epoch:02d}-{val_loss:.2f}.hdf5' 
model_save_best = 'weights/reg_multi-best-{epoch:02d}-{val_loss:.2f}.hdf5'  

#####################################

# customized callback class
class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

#############################################################################################
with tf.device('/cpu:0'):
    model = ResNet101(include_top=True, l2_norm = True, scale_param = 100, weights=None, classes = nb_classes) #weights=None for random initialization


# load weights
if saved_weights is not None:
    assert os.path.isfile(saved_weights)
    model.load_weights(saved_weights)


multi_model = multi_gpu_model(model, gpus = nb_gpus)
sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)
multi_model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) ############# PARAMS NOT FINALIZED#####


# generators
train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), n_classes = nb_classes, source = '/lfs2/tmp/anh-train/', mean_file = mean_img_file )
val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), n_classes = nb_classes, source = '/lfs2/tmp/anh-train/', mean_file = mean_img_file )

# callbacks
check_point = MultiGPUCheckpointCallback(filepath = model_save_path,base_model = model,  save_best_only=False,period = 10)   
check_point_best = MultiGPUCheckpointCallback(filepath = model_save_best, base_model = model, save_best_only=True)


#H = model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = val_datagen, validation_steps = 1000, callbacks =[csv_logger, check_point, check_point_best]  , workers = 8, use_multiprocessing=True)
H = multi_model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = val_datagen, validation_steps = 1000, callbacks =[csv_logger, check_point, check_point_best])
import numpy as np
np.save('logs/h1.npy',H)
