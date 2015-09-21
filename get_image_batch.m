function [ims, idxs] = get_image_batch(images, varargin)
%GET_IMAGE_BATCH  Load, preprocess, and pack images for CNN evaluation
%   images::
%       cell array of image paths 
%   `imageSize`:: [224, 224]
%       output size
%   `border`:: [0, 0]
%       border (within image)
%   `averageImage`:: []
%       average image that will be substracted (if not empty)
%   `interpolation`:: 'biliner'
%       resizing method
%   `keepAspect`:: true
%       whether keep the aspect ratio of input image
%   `augmentation`:: 'none'
%       specifies the operations (fliping, perturbation, etc.) used 
%       to get sub-regions
%       TODO: augmentation doesn't seem right, use 'none' or 'f2' only 
%   `invert`::false
%       change to true to map intensity value v-->255-v
%   `numThreads`:: 0
%       #threads for vl_imreadjpeg
%   `prefetch`:: false
%       prefetch option for vl_imreadjpeg
% 

% options in net.normalization
opts.imageSize = [224, 224] ;
opts.border = [0, 0] ;
opts.averageImage = [] ;
opts.interpolation = 'bilinear' ;
opts.keepAspect = true;
% other options
opts.augmentation = 'none' ;
opts.invert = false;
opts.numThreads = 0 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

switch opts.augmentation
  case 'none'
    tfs = [.5 ; .5 ; 0 ];
  case 'f2'
    tfs = [...
	0.5 0.5 ;
	0.5 0.5 ;
	  0   1];
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
end

nAugments = size(tfs,2);
nImages = numel(images); 

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = nImages > 1 && ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

im = cell(1, nImages) ;
if opts.numThreads > 0
  if prefetch
    vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
    ims = [] ;
    return ;
  end
  if fetch
    im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
  end
end
if ~fetch
  im = images ;
end

ims = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
            nImages*nAugments, 'single') ;

[~,augIdxs] = sort(rand(nAugments, nImages), 1) ;

for i=1:nImages

  % acquire image
  if isempty(im{i})
    imPath = strrep(images{i},'\',filesep);
    imt = imread(imPath) ;
    imt = single(imt) ; % faster than im2single (and multiplies by 255)
  else
    imt = im{i} ;
  end
  if opts.invert, 
    imt = 255 - imt;
  end
  if size(imt,3) == 1
    imt = cat(3, imt, imt, imt) ;
  end

  % resize
  w = size(imt,2) ;
  h = size(imt,1) ;
  factor = [(opts.imageSize(1)+opts.border(1))/h ...
            (opts.imageSize(2)+opts.border(2))/w];

  if opts.keepAspect
    factor = max(factor) ;
  end
  if any(abs(factor - 1) > 0.0001)
    imt = imresize(imt, ...
                   'scale', factor, ...
                   'method', opts.interpolation) ;
  end

  % crop & flip
  w = size(imt,2) ;
  h = size(imt,1) ;
  for ai = 1:nAugments
    t = augIdxs(ai,i) ;
    tf = tfs(:,t) ;
    dx = floor((w - opts.imageSize(2)) * tf(2)) ;
    dy = floor((h - opts.imageSize(1)) * tf(1)) ;
    sx = (1:opts.imageSize(2)) + dx ;
    sy = (1:opts.imageSize(1)) + dy ;
    if tf(3), sx = fliplr(sx) ; end
    ims(:,:,:,(ai-1)*nImages+i) = imt(sy,sx,:) ;
  end
end

if ~isempty(opts.averageImage)
  ims = bsxfun(@minus, ims, opts.averageImage) ;
end

% indexs of image sub-regions
idxs = reshape(repmat((1:nImages)',[1, nAugments]),[1 nAugments*nImages]);

