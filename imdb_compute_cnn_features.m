function imdb_compute_cnn_features( imdbName, modelName, varargin )
%IMDB_COMPUTE_CNN_FEATURES Compute and save CNN activations features 
% 
%   imdb:: 'Pf4q'
%       name of a folder under 'data/'
%   modelName:: 'imagenet-vgg-m'
%       model will be searched/saved under 'data/models'
%   `layers`:: {'fc6', 'fc7', 'fc8'}
%       the set of raw activations features that will be extracted
%   `augmentation`:: 'nr3'
%       1st field(f|n) indicates whether include flipped copy or not
%       2nd field(s|r) indicates type of region - Square or Rectangle
%       3rd field(1..4) indicates number of levels 
%       note: 'none', 'ns1', 'nr1' are equivalent
%   `gpuMode`:: false
%       set to true to compute on GPU 
%   `restart`:: false
%       set to true to re-compute all features 
%   `pca`:: 500
%       set to Inf to disable pca
%   `pcaNumSamples`:: Inf
%       set to a smaller value (e.g. 10^5) if pca takes too long 
%   `whiten`:: true
%       set to false to diable whitening 
%   `powerTrans`:: 2
%       set to 1 to disable power transform 

if nargin<2 || isempty(modelName), 
    modelName = 'imagenet-vgg-m';
end
if nargin<1 || isempty(imdbName), 
    imdbName = 'Pf4q';
end

% default options
opts.layers = {'fc6', 'fc7', 'fc8'};
opts.augmentation = 'nr3';
opts.gpuMode = false;
opts.restart = false;
opts.pca = 500;
opts.pcaNumSamples = Inf;
opts.whiten = true;
opts.powerTrans = 2;
opts = vl_argparse(opts,varargin);

% saving directory
saveDir = fullfile('data','features',[imdbName '-' modelName '-' ...
    opts.augmentation]);
if opts.restart, 
    rmdir(saveDir,'s');
end
cacheDir = fullfile(saveDir,'cache');
vl_xmkdir(cacheDir);

% data augmentation 
subWins = get_augmentation_matrix(opts.augmentation);
nSubWins = size(subWins,2);

% -------------------------------------------------------------------------
%                                                                 Get imdb
% -------------------------------------------------------------------------
imdb = get_imdb(imdbName);
nImgs = numel(imdb.images.name); 

% -------------------------------------------------------------------------
%                                                                CNN Model
% -------------------------------------------------------------------------
netFilePath = fullfile('data','models', [modelName '.mat']);
% download model if not found
if ~exist(netFilePath,'file'),
    fprintf('Downloading model (%s) ...', modelName) ;
    vl_xmkdir(fullfile('data','models')) ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', ...
        [modelName '.mat']), netFilePath) ;
    fprintf(' done!\n');
end
net = load(netFilePath);
% use gpu if requested and possible
if opts.gpuMode, 
    if gpuDeviceCount()==0, 
        fprintf('No supported gpu detected! ');
        reply = input('Continue w/ cpu mode? Y/N [Y]:','s');
        if ~isempty(reply) && reply~='Y', 
            return;
        end
        opts.gpuMode = false;
    else
        gpuDevice(1);
        net = vl_simplenn_move(net,'gpu');
    end
end
% find requested layers
layerNames = cell(1,length(net.layers));
for i=1:length(layerNames), 
    layerNames{i} = net.layers{i}.name;
end
[~,layers.index] = ismember(opts.layers,layerNames);
layers.index = layers.index + 1;
layers.name = opts.layers;

% -------------------------------------------------------------------------
%                                                    Get raw CNN responses
% -------------------------------------------------------------------------
% response dimensions
fprintf('Testing model (%s) ...', modelName) ;
im = single(imread('peppers.png'));
im_ = imresize(im,net.normalization.imageSize(1:2));
if opts.gpuMode, im_ = gpuArray(im_); end
res = vl_simplenn(net,im_); 
layers.sizes = zeros(3,numel(layers.name));
for i = 1:numel(layers.name), 
    layers.sizes(:,i) = reshape(size(res(layers.index(i)).x),[3,1]);
end
fprintf(' done!\n');

poolobj = gcp('nocreate');
if isempty(poolobj) || opts.gpuMode, 
    poolSize = 0; 
else
    poolSize = poolobj.NumWorkers;
end
parfor (i=1:nImgs, poolSize)
%  for  i=1:nImgs, % if no parallel computing toolbox 
    [imCat, imName, ~] = fileparts(imdb.images.name{i});
    if exist(fullfile(cacheDir, [imCat '_' imName '.mat']),'file'), 
        continue;
    end
    im = imread(fullfile(imdb.imageDir,imdb.images.name{i}));
    
    feat = get_cnn_activations( im, net, subWins, layers, ...
        'gpuMode', opts.gpuMode);
    parsave(fullfile(cacheDir,[imCat '_' imName '.mat']),feat);
    
    cacheFiles = dir(fullfile(cacheDir,'*.mat'));
    fprintf('[%4d/%4d] %s\n',length(cacheFiles),nImgs,...
        fullfile(imdb.imageDir,imdb.images.name{i}));
    
end

% -------------------------------------------------------------------------
%                               Construct feature descriptors and encoders
% -------------------------------------------------------------------------
featCell = cell(1,numel(layers.name));
for fi=1:numel(layers.name), 
    if strcmp(layers.name{fi}(1:2),'fc'), % fully connected layers
        featCell{fi}.x = zeros(nImgs*nSubWins,layers.sizes(3,fi));
        featCell{fi}.id = reshape(repmat([1:nImgs],[nSubWins,1]),...
            [nImgs*nSubWins,1]);
        featCell{fi}.imdb = imdb;
        featCell{fi}.modelName = modelName;
        featCell{fi}.layerName = layers.name{fi};
    else
        error('feature type (%s) not yet supported.', layers.name{fi});
    end
end
fprintf('Loading raw features: \n');
for i=1:nImgs, 
    if mod(i,10)==0, fprintf('.'); end
    if mod(i,500)==0, fprintf(' %4d/%4d\n', i,nImgs); end
    [imCat, imName, ~] = fileparts(imdb.images.name{i});
    feat = load(fullfile(cacheDir,[imCat '_' imName '.mat']));
    for fi = 1:numel(layers.name), 
        featCell{fi}.x((i-1)*nSubWins+(1:nSubWins),:) = ...
            squeeze(feat.(layers.name{fi}))';
    end
end
fprintf(' %4d/%4d done! \n', nImgs,nImgs); 

nSamples = opts.pcaNumSamples;
if nSamples >= nImgs*nSubWins, 
    nSamples = nImgs*nSubWins;
end
fprintf('Normalization: \n');
for fi = 1:numel(layers.name), 
    featOldDim = size(featCell{fi}.x,2);
    % TODO deal with singlularity
    if featOldDim>nSamples, 
        error('No enough samples');
    end
    featNewDim = opts.pca;
    if featNewDim>=featOldDim, 
        featNewDim = featOldDim;
    end
    fprintf('[%d/%d] (%s): ', fi, numel(layers.name), layers.name{fi});
    fprintf('l2 ... ');
    featCell{fi}.x = bsxfun(@rdivide,featCell{fi}.x,...
        sqrt(sum(featCell{fi}.x.^2,2)));
    fprintf('pca (%d-D=>%d-D,N=%d,whiten=%d) ... ', featOldDim, ...
        featNewDim, nSamples, opts.whiten);
    roIdx = randperm(nImgs*nSubWins);
    x = featCell{fi}.x(roIdx(1:nSamples),:);
    featCell{fi}.pcaMean = mean(x);
    [coeff,~,latent] = pca(x);
    if opts.whiten, 
        coeff = bsxfun(@rdivide,coeff,sqrt(latent)');
    end
    featCell{fi}.pcaCoeff = coeff(:,1:featNewDim);
    featCell{fi}.x = bsxfun(@minus,featCell{fi}.x,featCell{fi}.pcaMean) ...
        * featCell{fi}.pcaCoeff;
    fprintf('l2 ... ');
    featCell{fi}.x = bsxfun(@rdivide,featCell{fi}.x,...
        sqrt(sum(featCell{fi}.x.^2,2)));
    featCell{fi}.powerTrans = opts.powerTrans;
    if featCell{fi}.powerTrans~=1, 
        fprintf('pow (%.1f) ... ',featCell{fi}.powerTrans);
        featCell{fi}.x = (abs(featCell{fi}.x).^featCell{fi}.powerTrans)...
            .* sign(featCell{fi}.x);
    end
    fprintf('done! \n');
end

% write to disk
fprintf('Saving feature descriptors & encoders: ');
for fi = 1:numel(layers.name), 
    fprintf('%s ... ',layers.name{fi});
    feat = featCell{fi};
    save(fullfile(saveDir,[layers.name{fi} '.mat']),'-struct','feat');
end
fprintf('done! \n');

end