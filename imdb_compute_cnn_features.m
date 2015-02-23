function featCell = imdb_compute_cnn_features( imdbName, model, varargin )
%IMDB_COMPUTE_CNN_FEATURES Compute and save CNN activations features
%
%   imdb:: 'Pf4q'
%       name of a folder under 'data/'
%   model:: 'imagenet-vgg-m'
%       can be either string (model name) or the actual net model
%       model will be searched/saved under 'data/models'
%   `layers`:: {'fc6', 'fc7', 'prob'}
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
%   `readOp`:: @imread_255
%       the operator that reads data from file
%   `normalization`:: true
%       set to false to turn off all normalization (incl. pca, whitening,
%       powerTrans)
%   `pca`:: 500
%       set to Inf to disable pca
%   `pcaNumSamples`:: Inf
%       set to a smaller value (e.g. 10^5) if pca takes too long
%   `whiten`:: true
%       set to false to diable whitening
%   `powerTrans`:: 2
%       set to 1 to disable power transform

if nargin<2 || isempty(model),
    model = 'imagenet-vgg-m';
end
if nargin<1 || isempty(imdbName),
    imdbName = 'Pf4q';
end
if ischar(model), 
    modelName = model; 
    net = [];
else
    modelName = 'NoName';
    net = model;
end

% default options
opts.layers = {'fc6', 'fc7', 'prob'};
opts.augmentation = 'nr3';
opts.gpuMode = false;
opts.restart = false;
opts.readOp = @imread_255;
opts.normalization = true;
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
if isempty(net),
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
end
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
    if isfield(net.layers{i},'name'), 
        layerNames{i} = net.layers{i}.name;
    else
        layerNames{i} = 'na';
    end
end
[~,layers.index] = ismember(opts.layers,layerNames);
layers.index = layers.index + 1;
layers.name = opts.layers;

% -------------------------------------------------------------------------
%                                                    Get raw CNN responses
% -------------------------------------------------------------------------
% response dimensions
fprintf('Testing model (%s) ...', modelName) ;
nChannels = size(net.layers{1}.filters,3); 
im0 = zeros(net.normalization.imageSize(1), ...
    net.normalization.imageSize(2), nChannels, 'single') * 255; 
if opts.gpuMode, im0 = gpuArray(im0); end
res = vl_simplenn(net,im0);
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
    im = opts.readOp(fullfile(imdb.imageDir,imdb.images.name{i}),nChannels);
    
    if isfield(imdb.meta,'invert') && imdb.meta.invert, 
        im = 255 - im;
    end
    
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
    if strcmp(layers.name{fi}(1:2), 'fc') ... % fully connected layers
        || strcmp(layers.name{fi}, 'prob'), % output layer
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

if opts.normalization,
    fprintf('Normalization: \n');
    nSamples = opts.pcaNumSamples;
    if nSamples >= nImgs*nSubWins,
        nSamples = nImgs*nSubWins;
    end
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
end

% write to disk
fprintf('Saving feature descriptors & encoders: ');
for fi = 1:numel(layers.name),
    fprintf('%s ... ',layers.name{fi});
    feat = featCell{fi};
    save(fullfile(saveDir,[layers.name{fi} '.mat']),'-struct','feat','-v7.3');
end
fprintf('done! \n');

end
