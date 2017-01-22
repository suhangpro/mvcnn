This demo code shows an caffe implementation of MVCNN based on a 12-view setting.

1. The network is defined in mvccn_12view.prototxt, where we use the caffe 'Slice' layer to sperate views and "Eltwise" layer for max view-pooling. Please check the prototxt file for more details.

2. MVCNNDataLayer.py provides an example to prepare the input data layer for the network. It requires caffe compiled with python layer. one needs to change the load function based on the data folder organization. 