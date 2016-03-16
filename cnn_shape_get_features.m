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
%   `imListMask`:: true(1,numel(imList)) 
%   `batchSize`:: 1
%       batch size for gpu mode
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
%   `verbose`:: true 

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

if isempty(imList); feats = []; return; end

% default options
opts.saveRoot = fullfile('data','features'); 
opts.saveNames = {}; 
opts.imListMask = true(1,numel(imList)); 
opts.batchSize = 1;
opts.aug = 'none';
opts.gpus = [];
opts.numWorkers = 12;
opts.restart = false;
opts.readOp = @imread_255;
opts.verbose = true; 
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
        if opts.verbose, fprintf('Downloading model (%s) ...', modelName) ; end
        vl_xmkdir(fullfile('data','models')) ;
        urlwrite(fullfile('http://maxwell.cs.umass.edu/mvcnn-data/models', ...
            [modelName '.mat']), netFilePath) ;
        if opts.verbose, fprintf(' done!\n'); end
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
if opts.verbose, fprintf('Testing model (%s) ...', modelName) ; end
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
    assert(sz4==1, 'Incompatible network'); 
    if (sz1~=1 || sz2~=1), 
        warning('Feature %s will have spatial span: %d x %d', ...
            layers.name{i}, sz1, sz2);
    end 
    layers.sizes(:,i) = [sz1; sz2; sz3];
end
if opts.verbose, fprintf(' done!\n'); end

% -------------------------------------------------------------------------
%                                                             Usage mode 2 
% -------------------------------------------------------------------------
if  ~iscell(imList) || ~ischar(imList{1}),  
    feats = get_activations(imList, net, layers, nViews, subWins, ~isempty(opts.gpus));
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
if opts.verbose, fprintf('Loading pre-computed features ... '); end
for fi = 1:numel(layers.name),
    featPath = fullfile(opts.saveRoot,[layers.name{fi} '.mat']);
    if ~exist(featPath, 'file'), 
        flag_found = false;
        break;
    end
    if opts.verbose, fprintf('%s ... ', layers.name{fi}); end
    featCell{fi} = load(featPath);
end
if flag_found, 
    if opts.verbose, fprintf('all found! \n'); end
    feats = struct();
    for fi = 1:numel(layers.name),
        feats.(layers.name{fi}) = featCell{fi};
    end
    return;
else
    if opts.verbose, fprintf('all/some feature missing! \n'); end
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
shapeMask = opts.imListMask(1:nViews:end); 

if opts.numWorkers<=1 || ~isempty(opts.gpus), 
    poolSize = 0;
    i=1;
    while i<=nShapes, 
        nCurr = 1;
        if shapeMask(i) && ~exist(fullfile(cacheDir, [saveNames{i} '.mat']),'file'),
            nCurr = min(opts.batchSize, nShapes-i+1); 
            im = cell(1,nViews*nCurr);
            for v = 1:nViews*nCurr, 
                im{v} = opts.readOp(imList{(i-1)*nViews+v}, nChannels);
            end
            feat = get_activations(im, net, layers, nViews, subWins, ~isempty(opts.gpus));
            for j=i:i+nCurr-1,
                for featName = fields(feat)', 
                    f.(featName{1}) = feat.(featName{1})(:,:,:,(j-i)*nSubWins+(1:nSubWins)); 
                end
                save(fullfile(cacheDir, [saveNames{j} '.mat']),'-struct','f');
            end
        end
        if opts.verbose, 
            for j=i:i+nCurr-1, 
                if mod(j,10)==0, fprintf('.'); end
                if mod(j,500)==0, fprintf('\t [%3d/%3d]\n',j,nShapes); end
            end
        end
        i = i + nCurr; 
    end
    if opts.verbose, fprintf(' done!\n');  end
else
    poolObj = gcp('nocreate');
    if isempty(poolObj) || poolObj.NumWorkers<opts.numWorkers, 
        if ~isempty(poolObj), delete(poolObj); end
        poolObj = parpool(opts.numWorkers);
    end
    poolSize = poolObj.NumWorkers;
    if opts.verbose, parfor_progress(nShapes); end
    parfor (i=1:nShapes, poolSize)
    %  for  i=1:nShapes, % if no parallel computing toolbox
        if shapeMask(i) && ~exist(fullfile(cacheDir, [saveNames{i} '.mat']),'file'),
            im = cell(1,nViews);
            for v = 1:nViews, 
                im{v} = opts.readOp(imList{(i-1)*nViews+v}, nChannels);
            end
            feat = get_activations(im, net, layers, nViews, subWins, ~isempty(opts.gpus));
            parsave(fullfile(cacheDir, [saveNames{i} '.mat']),feat);
            % fprintf(' %s\n',imList{(i-1)*nViews+1});
        end
        if opts.verbose, parfor_progress(); end
    end
    if opts.verbose, parfor_progress(0); end
end

% -------------------------------------------------------------------------
%                                   Construct and save feature descriptors
% -------------------------------------------------------------------------
if numel(layers.name)>1, 
    feats = cell(1,numel(layers.name));
    for fi=1:numel(layers.name),
        feats{fi} = zeros(nShapes*nSubWins,layers.sizes(3,fi), ...
            layers.sizes(1,fi), layers.sizes(2,fi), 'single');
    end
else
    fi=1;
    feats = zeros(nShapes*nSubWins,layers.sizes(3,fi), ...
        layers.sizes(1,fi), layers.sizes(2,fi), 'single');
end
if opts.verbose, fprintf('Loading raw features: \n'); end
for i=1:nShapes,
    if shapeMask(i), 
        feat = load(fullfile(cacheDir, [saveNames{i} '.mat']));
        for fi = 1:numel(layers.name),
            if numel(layers.name)>1,
                feats{fi}((i-1)*nSubWins+(1:nSubWins),:,:,:) = ...
                    permute(feat.(layers.name{fi}), [4 3 1 2]); 
            else
                feats((i-1)*nSubWins+(1:nSubWins),:,:,:) = ...
                    permute(feat.(layers.name{fi}), [4 3 1 2]); 
            end
        end
    end
    if opts.verbose, 
        if mod(i,10)==0, fprintf('.'); end
        if mod(i,500)==0, fprintf('\t [%3d/%3d]\n', i,nShapes); end
    end
end
if opts.verbose, fprintf(' %4d/%4d done! \n', nShapes,nShapes); end

% write to disk
if opts.verbose, fprintf('Saving feature descriptors: ') ; end
for fi = 1:numel(layers.name),
    if opts.verbose, fprintf('%s ... ',layers.name{fi}); end
    if numel(layers.name)>1, 
        feat = feats{fi};
        save(fullfile(opts.saveRoot, [layers.name{fi} '.mat']), ...
            'feat', '-v7.3');
    else
        save(fullfile(opts.saveRoot, [layers.name{fi} '.mat']), ...
            'feats', '-v7.3');
    end
end
if opts.verbose, fprintf('done! \n'); end


% ------------------------------------------------------------------------------
function feat = get_activations(im, net, layers, nViews, subWins, gpuMode)
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

batchSize = size(im,4)/nViews; 

featCell = cell(1,numel(layers.name));
for fi = 1:numel(layers.name),
    featCell{fi} = zeros(layers.sizes(1,fi),layers.sizes(2,fi),...
        layers.sizes(3,fi),nSubWins*batchSize, 'single');
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
        featCell{fi}(:,:,:,ri:nSubWins:end) = single(gather(res(layers.index(fi)).x));
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

