import numpy as np
import sys
caffe_root = './caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import scipy.io as sio
import h5py
import os
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('mvcnn_Train.prototxt')
solver.net.copy_from('bvlc_reference_caffenet.caffemodel')
solver.solve()
