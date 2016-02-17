function net = cnn_shape(dataName, varargin)
%CNN_SHAPE Train an MVCNN on a provided dataset 
%
%   dataName:: 
%     must be name of a folder under data/
%   `baseModel`:: 'imagenet-matconvnet-vgg-m'
%     learning starting point
%   `fromScratch`:: false
%     if false, only the last layer is initialized randomly
%     if true, all the weight layers are initialized randomly
%   `numFetchThreads`::
%     #threads for vl_imreadjpeg
%   `aug`:: 'none'
%     specifies the operations (fliping, perturbation, etc.) used 
%     to get sub-regions
%   `viewpoolPos` :: 'relu5'
%     location of the viewpool layer, only used when multiview is true
%   `includeVal`:: false
%     if true, validation set is also used for training 
%   `useUprightAssumption`:: true
%     if true, 12 views will be used to render meshes, 
%     otherwise 80 views based on a dodecahedron
% 
%   `train` 
%     training parameters: 
%       `learningRate`:: [0.001*ones(1, 10) 0.0001*ones(1, 10) 0.00001*ones(1,10)]
%         learning rate
%       `batchSize`: 128
%         set to a smaller number on limited memory
%       `momentum`:: 0.9
%         learning momentum
%       `gpus` :: []
%         a list of available gpus
% 
% Hang Su

opts.networkType = 'simplenn'; % only simplenn is supported currently 
opts.baseModel = 'imagenet-matconvnet-vgg-m';
opts.fromScratch = false; 
opts.dataRoot = 'data' ;
opts.imageExt = '.png';
opts.numFetchThreads = 0 ;
opts.multiview = true; 
opts.viewpoolPos = 'relu5';
opts.useUprightAssumption = true;
opts.aug = 'stretch';
opts.pad = 32; 
opts.includeVal = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.multiview, 
  opts.expDir = sprintf('%s-ft-%s-%s-%s', ...
    opts.baseModel, ...
    dataName, ...
    opts.viewpoolPos, ...
    opts.networkType); 
else
  opts.expDir = sprintf('%s-ft-%s-%s', ...
    opts.baseModel, ...
    dataName, ...
    opts.networkType); 
end
opts.expDir = fullfile(opts.dataRoot, opts.expDir);
[opts, varargin] = vl_argparse(opts,varargin) ;

opts.train.learningRate = [0.001*ones(1, 10) 0.0001*ones(1, 10) 0.00001*ones(1,10)];
opts.train.numEpochs = numel(opts.train.learningRate); 
opts.train.momentum = 0.9; 
opts.train.batchSize = 5; 
opts.train.gpus = []; 
opts.train = vl_argparse(opts.train, varargin) ;

if ~exist(opts.expDir, 'dir'), vl_xmkdir(opts.expDir) ; end

assert(strcmp(opts.networkType,'simplenn'), 'Only simplenn is supported currently'); 

% -------------------------------------------------------------------------
%                                                             Prepare data
% -------------------------------------------------------------------------
imdb = get_imdb(dataName, ...
  'func', @(s) setup_imdb_modelnet(s, ...
    'useUprightAssumption', opts.useUprightAssumption,...
    'ext', opts.imageExt), ...
  'rebuild', true);
if ~opts.multiview, 
  nViews = 1;
else
  nShapes = length(unique(imdb.images.sid));
  nViews = length(imdb.images.id)/nShapes;
end
imdb.meta.nViews = nViews; 

opts.train.train = find(imdb.images.set==1);
opts.train.val = find(imdb.images.set==2); 
if opts.includeVal, 
  opts.train.train = [opts.train.train opts.train.val];
  opts.train.val = [];
end
opts.train.train = opts.train.train(1:nViews:end);
opts.train.val = opts.train.val(1:nViews:end); 

% -------------------------------------------------------------------------
%                                                            Prepare model
% -------------------------------------------------------------------------
net = cnn_shape_init(imdb.meta.classes, ...
  'base', opts.baseModel, ...
  'restart', opts.fromScratch, ...
  'nViews', nViews, ...
  'viewpoolPos', opts.viewpoolPos, ...
  'networkType', opts.networkType);  

% -------------------------------------------------------------------------
%                                                                    Learn 
% -------------------------------------------------------------------------
switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

net = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train) ;

% -------------------------------------------------------------------------
%                                                                   Deploy
% -------------------------------------------------------------------------
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
bopts.numThreads = opts.numFetchThreads ;
bopts.pad = opts.pad; 
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;
bopts.transformation = opts.aug ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case 'dagnn'
    error('dagnn version not yet implemented');
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
if nargout > 1, labels = imdb.images.class(batch); end
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
nViews = imdb.meta.nViews; 

batch = bsxfun(@plus,repmat(batch(:)',[nViews 1]),(0:nViews-1)');
batch = batch(:)'; 

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;

if ~isVal, % training
  im = cnn_get_batch(images, opts, 'prefetch', nargout == 0); 
else
  im = cnn_get_batch(images, opts, 'prefetch', nargout == 0, ...
    'transformation', 'none'); 
end

