# Multi-view CNN (MVCNN) for shape recognition

[Project Page](http://vis-www.cs.umass.edu/mvcnn/)
![MVCNN pipeline](http://vis-www.cs.umass.edu/mvcnn/images/mvcnn.png)

The goal of the project is to learn a general purpose descriptor for shape recognition. To do this we train discriminative models for shape recognition using convolutional neural networks (CNNs) where view-based shape representations are the only cues. Examples include **line-drawings**, **clip art images where color is removed**, or **renderings of 3D models** where there is little or no texture information present. 

If you use any part of the code from this project, please cite:

  @inproceedings{su15mvcnn,
  author    = {Hang Su and Subhransu Maji and Evangelos Kalogerakis and Erik G. Learned{-}Miller},
  title     = {Multi-view convolutional neural networks for 3d shape recognition},
  booktitle = {Proc. ICCV}, 
  year      = {2015}}

## Other implementations

(These are implementations provided by friends or found online, and are listed here for your convenience. I do not provide direct support on them.)

* PyTorch (from my UMass labmate @jongchyisu): [mvcnn_pytorch](https://github.com/jongchyisu/mvcnn_pytorch)
* Caffe (from my UMass labmate @brotherhuang): Check out the [caffe](https://github.com/suhangpro/mvcnn/tree/master/caffe) folder
* Tensorflow (from @WeiTang114): [MVCNN-Tensorflow](https://github.com/WeiTang114/MVCNN-TensorFlow)
* Torch (from @eriche2016): [mvcnn.torch](https://github.com/eriche2016/mvcnn.torch)
* PyTorch (from @RBirkeland): [MVCNN-ResNet](https://github.com/RBirkeland/MVCNN-ResNet)

## Installation

* Install dependencies
```bash 
git submodule update --init
```

* Compile

compile for CPU: 
```bash 
# two environment variables might need to be set, e.g. MATLABDIR=<MATLAB_ROOT> MEX=<MATLAB_ROOT>/bin/mex
matlab -nodisplay -r "setup(true);exit;"
```
compile for GPU (w/ cuDNN): 
```bash
# 1) two environment variables might need to be set, e.g. MATLABDIR=<MATLAB_ROOT> MEX=<MATLAB_ROOT>/bin/mex
# 2) other compilation options (e.g. 'cudaRoot',<CUDA_ROOT>,'cudaMethod','nvcc','cudnnRoot',<CUDNN_ROOT>) 
#  might be needed in the 'struct(...)' as well depending on you system settings
matlab -nodisplay -r "setup(true,struct('enableGpu',true,'enableCudnn',true));exit;"
```
**Note**: you can alternatively run directly the scripts from the Matlab command window, e.g. for Windows installations:
setup(true,struct('enableGpu',true,'cudaRoot','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0','cudaMethod','nvcc'));
You may also need to add Visual Studio's cl.exe in your PATH environment (e.g., C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64)

## Usage

* Extract descriptor for a shape (.off/.obj mesh). The descriptor will be saved in a .txt file (e.g. bunny_descriptor.txt). Uses default model with no fine-tuning. Assumes upright orientation by default. 

```matlab
MATLAB> shape_compute_descriptor('bunny.off');
```

* Extract descriptor for all shapes in a folder (.off/.obj meshes). The descriptors will be saved in .txt files in the same folder. Assumes no upright orientation. 

```matlab
MATLAB> shape_compute_descriptor('my_mesh_folder/','useUprightAssumption',false);
```

* Extract descriptors for all shapes in a folder (.off/.obj meshes) and post-process descriptors with learned metric. Uses non-default models. 

```matlab
MATLAB> shape_compute_descriptor('my_mesh_folder/', 'cnnModel', 'my_cnn.mat', ...
'metricModel', 'my_metric.mat','applyMetric',true); 
```

* Download datasets for training/evaluation (should be placed under data/)
    * modelnet40v1 (12 views w/ upright assumption): [tarball](http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1.tar) (204M)
    * modelnet40v2 (80 views w/o upright assumption): [tarball](http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v2.tar) (1.3G)
    * shapenet55v1 (12 views w/ upright assumption): [tarball](http://maxwell.cs.umass.edu/mvcnn-data/shapenet55v1.tar) (2.4G)
    * shapenet55v2 (80 views w/o upright assumption): [tarball](http://maxwell.cs.umass.edu/mvcnn-data/shapenet55v2.tar) (15G)

* Run training examples (see run_experiments.m for details)
```bash
# LD_LIBRARY_PATH might need to be set, e.g. LD_LIBRARY_PATH=<CUDA_ROOT>/lib64:<CUDNN_ROOT> 
matlab -nodisplay -r "run_experiments;exit;"
```
