# CNNs for shape recognition

The goal of the project is to learn a general purpose descriptor for shape recognition. To do this we train discriminative models for image recognition using covolutional neural networks (CNNs) where shape is the only cue. Examples include line-drawings, clip art images where color is removed, or **renderings of 3D models** where there is little texture information present. 

## Installation and Usage

* install dependencies
``` 
#!bash
git submodule init
git submodule update
```
* compile

(1) compile for CPU
``` 
#!bash
MEX=<MATLAB_ROOT>/bin/mex matlab -nodisplay -r "setup(true);exit;"
```
(2) compile for GPU: 
``` 
#!bash
MEX=<MATLAB_ROOT>/bin/mex matlab -nodisplay -r "setup(true,struct('enableGpu',true));exit;"
```
(3) compile with cuDNN support: 
``` 
#!bash
MEX=<MATLAB_ROOT>/bin/mex matlab -nodisplay -r "setup(true,struct('enableGpu',true,
'cudaRoot',<CUDA_ROOT>,'cudaMethod','nvcc','enableCudnn',true,'cudnnRoot',<CUDNN_ROOT>));exit;"
```
* download datasets 
```
#!bash
#modelnet40 (286M)
cd data
wget http://pegasus.cs.umass.edu/deep-shape-data/modelnet40phong.tar
tar xf modelnet40phong.tar

#sketch (211M)
cd data
wget http://pegasus.cs.umass.edu/deep-shape-data/sketch160.tar
tar xf sketch160.tar

#clipart (701M)
cd data
wget http://pegasus.cs.umass.edu/deep-shape-data/clipart100.tar
tar xf clipart100.tar
```
* run experiments
```
#!bath
LD_LIBRARY_PATH=<CUDA_ROOT>/lib64:<CUDNN_ROOT> matlab -nodisplay -r "run_experiments;exit;"
```
*LD_LIBRARY_PATH* may not be necessary depending on your installation, e.g. whether includes cuDNN support. 

## **NOTE**
Information below this point may be outdated. Please check [our arXiv paper](http://arxiv.org/abs/1505.00880) for additional details. 

## Datasets

### Collecting a new dataset of shapes

#### Motivation ####
Current datasets for shape recognition are rather limited. The largest shape datasets out there ?? contain only a hundred categories. 

#### Leveraging clip-art on the web ###
We leverage large amounts of clip-art images available on the web to build large dataset of shape classes. Clip-art allow us to extract high quality boundaries of objects using simple computer vision techniques such a GrabCut and Global probability of boundary (gPb), which would otherwise not be possible from real images due to lack of contrast between object and background. Annotating such boundaries would be prohibitively expensive and time consuming. For example even the largest segmentation datasets such as Microsoft COCO have imprecise object boundaries. Morevoer, instances in such images to occlusion and truncation hence a large part of the object boundary many not reflect the object's shape. 

Clip-art images allows us to systematically study the role of external contours or sihouttles, and internal contours for shape recognition. Morevoer clip-art images tend to be higher quality unlike hand drawings and reflect the true shape of the object more. There is also a wide range of categories for which clip-art is available, unlike 3D object repositories which have good coverage for man-made objects but not organic categories such as animals and plants. One drawback of clip-art is that instances may be exxagerated in their cartoon like appearance, but we hope to prune such categories during the dataset collection stage. 

We start by collecting 1000 _clipart_ images for various categories which are likely to be recognizable by shape. Each of these are manually inspected to see if (a) they contain the correct class (b) are clutter-free and non-photorealistic (so that GrabCut and gPb are likely to work) (c) contain a single, non-truncated object. We use an interface where a grid of images are shown and good ones are selected. It takes less than a second to process each image, making this pipeline fast.


### Human sketch dataset

The human sketch dataset (Eitz et al, 2012) which contains 20,000 hand-drawn images of 250 categories such as airplanes, apples, bridges, etc. The accuracy of humans in recognizing these hand-drawings in only 73% for a number of resons including the quality of the sketches drawn. In a subsequent paper by Schneider and Tuytelaars (Siggraph Asia 2014) cleaned up the data by removing instances that humans find hard to recognize.

The current state of the art is **67.6%** accuracy on the _sketch_ dataset and **79.0%** accuracy on the _sketch-clean_ dataset evaluated on a subset containing 160 categories and 56 images per-category. We follow the same training and test protocol of Eitz et al. The best performance is achieved using SIFT Fisher vectors with spatial pyramid pooling and linear SVMs. 

### Swedish leaf dataset

### 3d shape datasets

## Results

In addition to experimenting with ImageNet pretrained models, we optionally fine-tune the models on the datasets isself. We report results using R-CNN where features are extracted from the penultimate layer, and D-CNN where Fisher vectors are constructed from filter banks extracted from the last convolutional layer (see reference below).

### Modelnet

 dataset (measure) | finetune| fc7 | fc7-vd 
 :---- | :---: | :---: | :---: | 
 modelnet10toon (acc) | - | 90.97% | 90.42% 
 modelnet10toon (acc) | modelnet10toon | **91.30%** | - 
 modelnet10toonedge (acc) | - | 90.64% | 90.53% 
 modelnet10toonedge (acc) | modelnet10toonedge | 90.31% | - 
 modelnet40toon (acc) | - | 90.40% | 89.95% 
 modelnet40toon (acc) | modelnet40toon | 90.24% | 89.95% 
 modelnet40toonedge (acc) | - | 89.26% | 89.75% 
 modelnet40toonedge (acc) | modelnet40toonedge | 90.28% | **91.13%** 
 
### Others
 dataset (measure) | finetune| fc7 | dcnn | dcnn-sp | fc7-vd | dcnn-vd | dcnn-vd-sp
 :---- | :---: | :---: | :---: | :---: | :---: | :---: |
 sketch (acc) | - | 63.4 % | 65.3 % | 65.3 % | 52.4 % | 67.8 % | 67.5 % 
 sketch (acc) | sketch | 68.6 % | 66.6 % | - | **73.1 %** | - | -  
 sketch (acc) | clipart-gpb | 64.8 % | - | - | 61.1 % | - | -  
 sketch-clean (acc) | - | 77.34% | - | - | 69.29% | - | -  
 sketch-clean (acc) | *sketch-clean* | 84.04% | - | - | **86.31%** | - | -  
 sketch-clean (acc) | clipart | 78.84% | - | - | - | - | -  
 sketch-clean (acc) | modelnet40toonedge | 79.35% | - | - | 77.94% | - | -  
 clipart (acc) | - | 72.35% | - | - | 69.57% | - | -  
 clipart (acc) | clipart | **79.91%** | - | - | - | - | -  
 clipart (acc) | modelnet40toon | 71.66% | - | - | 66.82% | - | -  
 clipart-gpb (acc) | - | 62.71% | - | - | 52.9 % | - | -  
 clipart-gpb (acc) | *clipart-gpb* | 71.90% | - | - | **76.11%** | - | -  

*italic*: fine-tuned with 50% dropout on fc6 and fc7

## Reference

For details on D-CNN read the following paper:

	@article{DBLP:journals/corr/CimpoiMV14,
  	author    = {Mircea Cimpoi and Subhransu Maji and Andrea Vedaldi},
  	title     = {Deep convolutional filter banks for texture recognition and segmentation},
  	journal   = {CoRR},
  	volume    = {abs/1411.6836},
 	year      = {2014},
  	url       = {http://arxiv.org/abs/1411.6836}}