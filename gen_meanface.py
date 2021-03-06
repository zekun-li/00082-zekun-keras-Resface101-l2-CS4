# -*- coding: utf-8 -*-
'''
convert caffe ResFace101 model for Keras.
save converted model+weights to 'model/keras_model.hdf5',
save mean_file to 'model/keras_mean_img.npy'

'''

import numpy as np
import warnings
import os


if __name__ == '__main__':

    import sys
    sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/sources/caffe-rc4_AS_l2norm/python')
    import caffe
    
    #mean_file = '/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran-train-layer_v2_clean_mow_mp_mgpu_n3_L2S100_poseCNN_bigset17/expts/trained_CNN/average_face.bin' #mean_face.jpg
    #mean_file = '/nfs/isicvlnas01/projects/glaive/expts//00036-iacopo-train-layer_v2_clean_mow_mp_mgpu_n3_more_aug_l2norm/expts/trained_CNN/average_face.bin' #mean_face_moreaug.jpg
    #mean_file = '/nfs/isicvlnas01/projects/glaive/expts//00036-iacopo-train-alexnet_bumpmaps_2/expts/average_face.bin' #mean_face_bumpmaps.jpg
    #mean_file = '/nfs/isicvlnas01/projects/glaive/expts//00036-iacopo-train-aug/expts/average_face.bin' #mean_face1.jpg
    mean_file = '/nfs/isicvlnas01/projects/glaive/expts//00036-iacopo-train-aug_resnet_noaug/expts/average_face.bin' #mean_face_noaug.jpg
    with open(mean_file) as f:
        mean_data = f.read()

    mean_blob = caffe.io.caffe_pb2.BlobProto.FromString(mean_data)
    mean_img = caffe.io.blobproto_to_array(mean_blob)[0]
    #with open('model/keras_mean_img.npy','w') as out_f:
    #    np.save(out_f, mean_img)
    import cv2
    cv2.imwrite('mean_face_noaug.jpg',mean_img.transpose(1,2,0))
    print ('done')
