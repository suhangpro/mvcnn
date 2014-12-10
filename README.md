## Convolutional Neural Networks for Shape Recognition

The goal of the project is to learn a general purpose shape descriptor for applications such as hand-drawn image classification, 3D object retrieval, etc. To do this we train discriminative models for image recognition using Covolutional Neural Networks (CNNs) where shape is the only cue. Examples include line-drawings, clip art, or renderings of 3D models where there is little texture information present. 


#### Collecting a dataset of shapes

We leverage clip-art datasets on the internet to build a large dataset of shape classes. Clip-art images enable us to extract high quality boundaries of objects using simple computer vision techniques such a GrabCut and Global probability of boundary (gPb). Annotating such boundaries would be prohibitively expensive and time consuming. 

We start by collecting 1000 clip-art images for various categories which are likely to be recognizable by shape. Each of these are manually inspected to see if (a) they contain the correct class (b) are clutter-free and non-photorealistic (so that GrabCut and gPb are likely to work) (c) contain a single, non-truncated object. We use an interface where a grid of images are shown and good ones are selected. It takes less than a second to process each image, making this pipeline fast. 

In addition we also experiment with the sketch dataset (Eitz et al, 2012) which contains 20,000 hand-drawn images of 250 categories such as airplanes, apples, bridges, etc. Due to the difficulty of sketching well, the accuracy of huamns in recognizing these hand-drawings in only 73%.

#### Fine-tine CNNs on the sketch dataset

Fine-tune the models trained on ImageNet classification challenge on the sketch dataset of Eitz et al., SIGGRAPH 2012. We replace the last layer of the CNNs with a 250 way classifier and continue training the model with a smaller learning rate. 

**TODO:** Fine-tune models on the shapes datset

#### R-CNN and D-CNN

We experiment with R-CNN and D-CNN, i.e., CNN filter banks with Fisher-vectors. 

	@article{DBLP:journals/corr/CimpoiMV14,
  	author    = {Mircea Cimpoi and Subhransu Maji and Andrea Vedaldi},
  	title     = {Deep convolutional filter banks for texture recognition and segmentation},
  	journal   = {CoRR},
  	volume    = {abs/1411.6836},
 	year      = {2014},
  	url       = {http://arxiv.org/abs/1411.6836}
	}

#### Source
The code is available on Bitbucket at [https://smaji@bitbucket.org/smaji/deep-shape.git](https://smaji@bitbucket.org/smaji/deep-shape.git)

#### Acknowlegements

The code uses open source implementations such as [matconvnet](http://www.vlfeat.org/matconvnet/) and [vlfeat](http://www.vlfeat.org).

#### Questions or comments	
_For questions or comments email Subhransu Maji (smaji@cs.umass.edu)_