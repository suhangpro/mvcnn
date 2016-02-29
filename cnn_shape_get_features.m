function feats = cnn_shape_get_features( imList, model, layers, varargin )
%CNN_SHAPE_GET_FEATURES Compute and save CNN activation features
%
%   imList:: 
%       mode 1: cell array of image paths
%       mode 2: cell array of images OR stacked image tensor
%   model:: 'imagenet-matconvnet-vgg-m'
%       can be either string (model name) or the actual net model
%       model will be searched/saved under 'data/models'
%   layers:: 
%       names of layers that will be used as feature
%   `saveRoot`:: 'data/features'
%       features will be saved under `saveRoot` in respective sub-folders
%   `saveNames`:: {}
%       containing names that will be used for output files
%   `aug`:: 'none'
%       1st field(f|n) indicates whether include flipped copy or not
%       2nd field(s|r) indicates type of region - Square or Rectangle
%       3rd field(1..4) indicates number of levels
%       note: 'none', 'ns1', 'nr1' are equivalent
%       note: `aug` can also be used to pass in sub-windows directly
%   `gpus`:: []
%       set to enable GPU
%       currently can only use 1 (the first if multiple is specified)
%   `numWorkers`:: 12
%       number of CPU workers, only in use when gpus is empty
%   `restart`:: false
%       set to true to re-compute all features
%   `readOp`:: @imread_255
%       the operator that reads data from file

if ~exist('model','var') || isempty(model),
    model = 'imagenet-matconvnet-vgg-m';
end
if ischar(model), 
    modelName = model; 
    net = [];
else
    modelName = 'NoName';
    net = model;
end

% default options
opts.saveRoot = fullfile('data','features'); 
opts.saveNames = {}; 
opts.aug = 'none';
opts.gpus = [];
opts.numWorkers = 12;
opts.restart = false;
opts.readOp = @imread_255;
[opts,varargin] = vl_argparse(opts,varargin);

% data augmentation
if ischar(opts.aug), 
    subWins = get_augmentation_matrix(opts.aug);
else
    subWins = opts.aug;
end
nSubWins = size(subWins,2);

% -------------------------------------------------------------------------
%                               CNN Model: net, nViews, nChannels, nShapes
% -------------------------------------------------------------------------
if isempty(net),
    netFilePath = fullfile('data','models', [modelName '.mat']);
    % download model if not found
    if ~exist(netFilePath,'file'),
        fprintf('Downloading model (%s) ...', modelName) ;
        vl_xmkdir(fullfile('data','models')) ;
        urlwrite(fullfile('http://maxwell.cs.umass.edu/mvcnn-data/models', ...
            [modelName '.mat']), netFilePath) ;
        fprintf(' done!\n');
    end
    net = load(netFilePath);
end

% use the first gpu if specified
if ~isempty(opts.gpus),
    gpuDevice(opts.gpus(1));
    net = vl_simplenn_move(net,'gpu');
end

% see if it's a multivew net
viewpoolIdx = find(cellfun(@(x)strcmp(x.name, 'viewpool'),net.layers));
if ~isempty(viewpoolIdx), 
    if numel(viewpoolIdx)>1, 
        error('More than one viewpool layers found!'); 
    end
    if ~isfield(net.layers{viewpoolIdx},'vstride'),  	
        nViews = net.layers{viewpoolIdx}.stride;
    else
        nViews = net.layers{viewpoolIdx}.vstride;
    end
else
    nViews = 1;
end

nShapes = numel(imList) / nViews; 

% -------------------------------------------------------------------------
%                                                          Response layers
% -------------------------------------------------------------------------
% response dimensions
fprintf('Testing model (%s) ...', modelName) ;
if isfield(net.layers{1},'weights'), 
  nChannels = size(net.layers{1}.weights{1},3); 
else
  nChannels = size(net.layers{1}.filters,3);  % old format
end
im0 = zeros(net.meta.normalization.imageSize(1), ...
    net.meta.normalization.imageSize(2), nChannels, nViews, 'single') * 255; 
if opts.gpus, im0 = gpuArray(im0); end
res = vl_simplenn(net,im0);
layers = struct('name', {layers}, 'sizes', [], 'index', []); 
for i = 1:numel(layers.name), 
    layers.index(i) = 1 + find(cellfun(@(c) strcmp(c.name, layers.name{i}), net.layers));
    [sz1, sz2, sz3, sz4] = size(res(layers.index(i)).x);
    assert(sz1==1 && sz2==1 && sz4==1); 
    layers.sizes(:,i) = [sz1; sz2; sz3];
end
fprintf(' done!\n');

% -------------------------------------------------------------------------
%                                                             Usage mode 2 
% -------------------------------------------------------------------------
if  ~iscell(imList) || ~ischar(imList{1}),  
    feats = get_activations(imList, net, layers, subWins, ~isempty(opts.gpus));
    return; 
end

% -------------------------------------------------------------------------
%                                                   Load data if available
% -------------------------------------------------------------------------
% saving directory
if opts.restart,
    rmdir(opts.saveRoot,'s');
end
cacheDir = fullfile(opts.saveRoot,'cache');
vl_xmkdir(cacheDir);

featCell = cell(1,numel(layers.name));
flag_found = true;
fprintf('Loading pre-computed features ... ');
for fi = 1:numel(layers.name),
    featPath = fullfile(opts.saveRoot,[layers.name{fi} '.mat']);
    if ~exist(featPath, 'file'), 
        flag_found = false;
        break;
    end
    fprintf('%s ... ', layers.name{fi});
    featCell{fi} = load(featPath);
end
if flag_found, 
    fprintf('all found! \n');
    feats = struct();
    for fi = 1:numel(layers.name),
        feats.(layers.name{fi}) = featCell{fi};
    end
    return;
else
    fprintf('all/some feature missing! \n');
    clear featCell;
end

% -------------------------------------------------------------------------
%                                                    Get raw CNN responses
% -------------------------------------------------------------------------
if ~isempty(opts.saveNames), 
    saveNames = opts.saveNames;
else
    saveNames = cellfun(@(s) get_name_str(s, nViews), ...
        imList(1:nViews:end), 'UniformOutput', false);
end

if opts.numWorkers<=1 || ~isempty(opts.gpus), 
    poolSize = 0;
    for  i=1:nShapes, 
        if ~exist(fullfile(cacheDir, [saveNames{i} '.mat']),'file'),
            im = cell(1,nViews);
            for v = 1:nViews, 
                im{v} = opts.readOp(imList{(i-1)*nViews+v}, nChannels);
            end
            feat = get_activations(im, net, layers, subWins, ~isempty(opts.gpus));
            save(fullfile(cacheDir, [saveNames{i} '.mat']),'-struct','feat');
        end
        if mod(i,10)==0, fprintf('.'); end
        if mod(i,500)==0, fprintf('\t [%3d/%3d]\n',i,nShapes); end
    end
    fprintf(' done!\n'); 
else
    poolObj = gcp('nocreate');
    if isempty(poolObj) || poolObj.NumWorkers<opts.numWorkers, 
        if ~isempty(poolObj), delete(poolObj); end
        poolObj = parpool(opts.numWorkers);
    end
    poolSize = poolObj.NumWorkers;
    parfor_progress(nShapes); 
    parfor (i=1:nShapes, poolSize)
    %  for  i=1:nShapes, % if no parallel computing toolbox
        if exist(fullfile(cacheDir, [saveNames{i} '.mat']),'file'),
            continue;
        end
        im = cell(1,nViews);
        for v = 1:nViews, 
            im{v} = opts.readOp(imList{(i-1)*nViews+v}, nChannels);
        end
        feat = get_activations(im, net, layers, subWins, ~isempty(opts.gpus));
        parsave(fullfile(cacheDir, [saveNames{i} '.mat']),feat);
        % fprintf(' %s\n',imList{(i-1)*nViews+1});
        parfor_progress();
    end
    parfor_progress(0);
end

% -------------------------------------------------------------------------
%                                   Construct and save feature descriptors
% -------------------------------------------------------------------------
feats = cell(1,numel(layers.name));
for fi=1:numel(layers.name),
    feats{fi} = zeros(nShapes*nSubWins,layers.sizes(3,fi), 'single');
end
fprintf('Loading raw features: \n');
for i=1:nShapes,
    if mod(i,10)==0, fprintf('.'); end
    if mod(i,500)==0, fprintf(' %4d/%4d\n', i,nShapes); end
    feat = load(fullfile(cacheDir, [saveNames{i} '.mat']));
    for fi = 1:numel(layers.name),
        feats{fi}((i-1)*nSubWins+(1:nSubWins),:) = ...
            squeeze(feat.(layers.name{fi}))';
    end
end
fprintf(' %4d/%4d done! \n', nShapes,nShapes);

% write to disk
fprintf('Saving feature descriptors: ');
for fi = 1:numel(layers.name),
    fprintf('%s ... ',layers.name{fi});
    feat = feats{fi};
    save(fullfile(opts.saveRoot, [layers.name{fi} '.mat']), ...
        '-struct', 'feat', '-v7.3');
end
fprintf('done! \n');


% ------------------------------------------------------------------------------
function feat = get_activations(im, net, layers, subWins, gpuMode)
% ------------------------------------------------------------------------------

nSubWins = size(subWins,2);
if isfield(net.layers{1},'weights'), 
  nChannels = size(net.layers{1}.weights{1},3);
else
  nChannels = size(net.layers{1}.filters,3); % old format
end

if iscell(im), 
    imCell = im;
    im = zeros(net.meta.normalization.imageSize(1), ...
        net.meta.normalization.imageSize(2), ...
        nChannels, ...
        numel(imCell));
    for i=1:numel(imCell), 
        if size(imCell{i},3) ~= nChannels, 
            error('image (%d channels) is not compatible with net (%d channels)', ...
                size(imCell{i},3), nChannels);
        end
        im(:,:,:,i) = imresize(imCell{i}, net.meta.normalization.imageSize(1:2));
    end
elseif size(im,3) ~= nChannels, 
    error('image (%d channels) is not compatible with net (%d channels)', ...
        size(im,3), nChannels);
end

featCell = cell(1,numel(layers.name));
for fi = 1:numel(layers.name),
    featCell{fi} = zeros(layers.sizes(1,fi),layers.sizes(2,fi),...
        layers.sizes(3,fi),nSubWins, 'single');
end

im = single(im);
averageImage = net.meta.normalization.averageImage;
if numel(averageImage)==nChannels, 
    averageImage = reshape(averageImage, [1 1 nChannels]); 
end

for ri = 1:nSubWins,
    r = subWins(1:4,ri).*[size(im,2);size(im,2);size(im,1);size(im,1)];
    r = round(r);
    im_ = im(max(1,r(3)):min(size(im,1),r(3)+r(4)),...
        max(1,r(1)):min(size(im,2),r(1)+r(2)),:,:);
    if subWins(5,ri), im_ = flipdim(im_,2); end
    im_ = bsxfun(@minus, imresize(im_, net.meta.normalization.imageSize(1:2)), ...
        averageImage);
    if gpuMode,
        im_ = gpuArray(im_);
    end
    res = vl_simplenn(net,im_);
    for fi = 1:numel(layers.name),
        featCell{fi}(:,:,:,ri) = single(gather(res(layers.index(fi)).x));
    end
end
% pack features into structure
feat = struct;
for fi = 1:numel(layers.name),
    feat.(layers.name{fi}) = featCell{fi};
end

function s = get_name_str(s, nv)
[~,s] = fileparts(s); 
if nv>1, 
    suffix_idx = strfind(s,'_');
    if ~isempty(suffix_idx),
        s = s(1:suffix_idx(end)-1);
    end
end

