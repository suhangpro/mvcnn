# Caffe data layer of generating moving mnist images

import caffe
import numpy as np
import math
import scipy
import sys
import os
import cv2
from PIL import Image
from random import shuffle

class MVCNNDataLayer(caffe.Layer):
  """Caffe moving mnist data layer used for training."""
  def setup(self, bottom, top):
    """Setup the MovingMnistDataLayer."""

    # parse the layer parameter string, which must be valid YAML
    layer_params = eval(self.param_str)
    self._dataPath = layer_params['data_path']
    self._batch_size = layer_params['batch_size']
    #self._mean_file = layer_params['mean_file']
    self._view_Size = layer_params['view_size']
    self._channel_Size = layer_params['channel_size']
    self._phase = layer_params['phase']   # train or test
    self._name_to_top_map = {'data': 0, 'label': 1}
    self._classList = next(os.walk(self._dataPath))[1]
    self._modelList = []
    self._model2lable = {}
    self._train_iteration = 0	
    for i in range(len(self._classList)):
        imageFiles = os.listdir(self._dataPath + '/' + self._classList[i] + '/' + self._phase)
        for image in imageFiles:
            modelName = image[0:-8]
	    self._model2lable[modelName] = i
            if self._modelList.count(modelName) == 0:
		self._modelList.append(modelName)
    top[0].reshape(self._batch_size * self._view_Size ,self._channel_Size, 227, 227)
    top[1].reshape(self._batch_size, 1, 1, 1)

  def forward(self, bottom, top):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()

    for blob_name, blob in blobs.iteritems():
      top_ind = self._name_to_top_map[blob_name]
      # Reshape net's input blobs
      top[top_ind].reshape(*(blob.shape))
      # Copy data into net's input blobs
      top[top_ind].data[...] = blob.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

  def _loadSampleImage(self, modelName):
    mean_file = np.load('ilsvrc_2012_mean.npy')
    pixelMeans = mean_file.mean(1).mean(1)
    imh = 227
    imw = 227
    imc = 3
    ims = np.zeros((imh, imw, imc, self._view_Size), dtype=np.float32)
    label = self._model2lable[modelName]
    for view in range(1, self._view_Size + 1):
      viewID = str(view)
      while len(viewID) < 3 : viewID = '0' + viewID
      img = self._dataPath + '/' + self._classList[label] + '/' + self._phase  + '/' + modelName + '_' + viewID + '.jpg'
      im = cv2.imread(img)
      assert(im is not None)
      im = cv2.resize(im, (imh, imw))
      im = im.astype(np.float32, copy=False) - pixelMeans 
      ims[:,:,:,view-1] = im
    ims = ims.transpose(3,2,0,1)
    return ims 

  def reshape(self, bottom, top):
    """Reshaping happens during the call to forward."""
    pass

  def _get_next_minibatch(self):
    height = 227
    width = 227
    batch_size = self._batch_size
    #if self._train_iteration == 0 : shuffle(self._modelList)
    data = np.ones((self._batch_size * self._view_Size, self._channel_Size, height, width), dtype=np.float32)
    label = np.ones((batch_size, 1, 1, 1), dtype=np.float32)
    for i in range(batch_size):
	currentModel = self._modelList[(self._train_iteration * self._batch_size + i) % len(self._modelList)] 
	#print currentModel
	currentImage = self._loadSampleImage(currentModel)
	for j in range(self._view_Size) :
		data[j * self._batch_size + i,:,:,:] = currentImage[j,:,:,:]
	label[i,:,:,:] = self._model2lable[currentModel]
    self._train_iteration += 1
    blobs = {'data': data, 'label': label}
    return blobs
