# CNNs for shape recognition

The goal of the project is to learn a general purpose descriptor for shape recognition. To do this we train discriminative models for image recognition using covolutional neural networks (CNNs) where shape is the only cue. Examples include line-drawings, clip art images where color is removed, or renderings of 3D models where there is little texture information present. 

## Installing and compiling

* install dependencies
``` 
#!bash
git submodule init
git submodule update
```
* compile
``` 
#!bash
MEX=MATLAB_DIR/bin/mex matlab -nodisplay -r "setup(true);exit;"
```
to compile with GPU support: 
``` 
#!bash
MEX=MATLAB_DIR/bin/mex matlab -nodisplay -r "setup(true,true);exit;"
```
* download datasets 
```
#!bash
#clipartgpb (316M)
cd data
wget http://pegasus.cs.umass.edu/deep-shape-data/clipartgpb.tar
tar xf clipartgpb.tar

#sketch (525M)
cd data
wget http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
mkdir sketch
unzip sketches_png.zip -d sketch/
```
## Datasets
![size comparison](https://dl.dropboxusercontent.com/u/50473730/tmp/dataset.png)

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

 dataset (measure) | finetune| fc7 | dcnn | dcnn-sp | fc7-vd | dcnn-vd | dcnn-vd-sp
 :---- | :---: | :---: | :---: | :---: | :---: | :---: |
 sketch (acc) | - | 63.4 % | 65.3 % | 65.3 % | 52.4 % | 67.8 % | 67.5 % 
 sketch (acc) | sketch | 68.6 % | 66.6 % | - | **73.1 %** | - | -  
 sketch (acc) | clipart-gpb | 64.8 % | - | - | 61.1 % | - | -  
 sketch-clean (acc/ mAP) | - | 77.34%/ 0.7798 | - | - | 69.29%/ 0.6765 | - | -  
 sketch-clean (acc/ mAP) | sketch-clean | 83.95%/ 0.8627 | - | - | **86.31%**/ **0.8893** | - | -  
 sketch-clean (acc/ mAP) | clipart | 78.84%/ 0.7997 | - | - | - | - | -  
 sketch-clean (acc/ mAP) | modelnet40toonedge | 79.35%/ 0.8126 | - | - | 77.94%/ 0.7853 | - | -  
 clipart (acc/ mAP) | - | 72.35%/ 0.7283 | - | - | 69.57%/ 0.6981 | - | -  
 clipart (acc/ mAP) | clipart | **79.91%**/ **0.8180** | - | - | - | - | -  
 clipart (acc/ mAP) | modelnet40toon | 71.66%/ 0.7267 | - | - | 66.82%/ 0.6764 | - | -  
 clipart-gpb (acc) | - | 62.7 % | - | - | 52.9 % | - | -  
 clipart-gpb (acc) | clipart-gpb | 67.4 % | - | - | **77.2 %** | - | -  
 modelnet10toon (acc/ mAP) | modelnet10toon | **90.97%**/ **0.9570** | - | - | - | - | -  
 modelnet10toonedge (acc/ mAP) | - | 89.98%/ 0.9558 | - | - | - | - | -  
 modelnet10toonedge (acc/ mAP) | modelnet10toonedge | 90.75%/ 0.9566 | - | - | - | - | -  
 modelnet40toon (acc/ mAP) | - | 90.40%/ 0.9146 | - | - | 89.95%/ 0.9112 | - | -  
 modelnet40toon (acc/ mAP) | modelnet40toon | 90.84%/ 0.9167 | - | - | **91.05%**/ 0.9224 | - | -  
 modelnet40toonedge (acc/ mAP) | - | 89.26%/ 0.9109 | - | - | 89.75%/ 0.9108 | - | -  
 modelnet40toonedge (acc/ mAP) | modelnet40toonedge | 90.28%/ 0.9170 | - | - | 90.64%/ **0.9229** | - | -  
 
## Reference

For details on D-CNN read the following paper:

	@article{DBLP:journals/corr/CimpoiMV14,
  	author    = {Mircea Cimpoi and Subhransu Maji and Andrea Vedaldi},
  	title     = {Deep convolutional filter banks for texture recognition and segmentation},
  	journal   = {CoRR},
  	volume    = {abs/1411.6836},
 	year      = {2014},
  	url       = {http://arxiv.org/abs/1411.6836}}