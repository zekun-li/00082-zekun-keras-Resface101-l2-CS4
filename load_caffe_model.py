import os
import sys
import numpy as np
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/sources/caffe-rc4_AS_l2norm/python')
#sys.path.insert(0,'/nfs/isicvlnas01/users/iacopo/sources/caffe-Sep-2016_py_cpu/python')
import caffe
caffe.set_device(0)
net  = caffe.Net('/nfs/isicvlnas01/users/iacopo/cnn_models/resnet-101_new_augm_mow/ResNet-101-deploy_augmentation_l2scale_alpha_100.prototxt','/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran-train-layer_v2_clean_mow_mp_mgpu_n3_L2S100_poseCNN_bigset17/expts/trained_CNN/snap_resnet__iter_285000.caffemodel', caffe.TEST)

'''
I1204 16:51:56.385507 26555 net.cpp:244] This network produces output pool5_l2_s
I1204 16:52:00.961480 26555 net.cpp:746] Ignoring source layer data
I1204 16:52:01.366075 26555 net.cpp:746] Ignoring source layer fc_ft_aug_newlayer_scale_v1
I1204 16:52:01.366159 26555 net.cpp:746] Ignoring source layer loss

Cannot copy param 0 weights from layer 'fc_ft_aug_newlayer_scale_v1'; shape mismatch.  Source param shape is 62955 2048 (128931840); target param shape is 68465 2048 (140216320). To learn this layer's parameters from scratch rather than copying from a saved net, rename the layer.


'''

for name, data in net.params.iteritems():
    #print name
    #for i in range(len(data)):
    print name, '\t\t',[data[i].data.shape for i in range(len(data))]

#len(net.layers) = 483
#weight = net.params['fc_ft_augm3'][0].data 
'''
bias = net.params['fc_ft_augm3'][1].data
weight = weight.transpose(1,0)
weights = [weight, bias]
import cPickle as pickle
with open("small_caffe_clf_weights.pkl","w") as f:
    pickle.dump(weights,f)
'''
np.random.seed(123)
x = np.random.randint(0,256, size = ( 3,224,224))
#x = np.expand_dims(x, axis=0)
net.blobs['data'].data[...] = x
print net.forward()


print 'done'




