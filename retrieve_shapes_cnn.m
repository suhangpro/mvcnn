function [ results ] = retrieve_shapes_cnn( shape, feat, varargin )
%RETRIEVE_SHAPES_CNN Retrieve 3d shapes using CNN features
%
%   shape::
%       Can be either
%         cell array containing projected images or paths to them
%       OR
%         [] (evaluation within dataset)
%   feat::
%       structure containing features from reference images and meta data
%       .x
%       .id
%       .imdb
%       .modelName
%       .layerName
%       .pcaMean
%       .pcaCoeff
%       .powerTrans
%   `method`:: 'mindist'
%       other choices: 'avgdist', 'avgdesc'
%   `net`::[]
%       preloaded cnn model (required when feat.modelName is not available)
%   `gpuMode`:: false
%       set to true to compute on GPU
%   `nTop`:: Inf
%       number of images in results
%   `querySets`:: {'test'}
%       set of query shapes (used only when shape==[])
%   `refSets`:: {'test'}
%       set of reference images
%   `metric`:: 'L2'
%       other choices: 'LINF', 'L1', 'L0', 'CHI2', 'HELL'
%
% Hang Su

% default options
opts.method = 'mindist';
opts.net = [];
opts.gpuMode = false;
opts.nTop = Inf;
opts.querySets = {'test'};
opts.refSets = {'test'};
opts.metric = 'L2';
opts = vl_argparse(opts,varargin);

% sort and group images according to shape id
imdb = feat.imdb;
[imdb.images.sid, order] = sort(imdb.images.sid);
imdb.images.name = imdb.images.name(order);
imdb.images.class = imdb.images.class(order);
imdb.images.set = imdb.images.set(order);
imdb.images.id = imdb.images.id(order);
feat.x = feat.x(order,:);
nRefViews = length(imdb.images.name)/imdb.meta.nShapes; % NOTE: assume same # of views
shapeGtClasses = imdb.images.class(1:nRefViews:end);
shapeSets = imdb.images.set(1:nRefViews:end);

[~,I] = ismember(opts.refSets,imdb.meta.sets);
refImgInds = ismember(imdb.images.set,I);
refShapeIds=find(ismember(shapeSets,I));
nRefShapes = numel(refShapeIds);
refX = feat.x(refImgInds, :);
nDims = size(refX,2);

if ~isempty(shape), % retrieval given a query shape
    % load model
    net = opts.net;
    if isempty(net),
        netFilePath = fullfile('data','models', [feat.modelName '.mat']);
        % download model if not found
        if ~exist(netFilePath,'file'),
            fprintf('Downloading model (%s) ...', feat.modelName) ;
            vl_xmkdir(fullfile('data','models')) ;
            urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', ...
                [feat.modelName '.mat']), netFilePath) ;
            fprintf(' done!\n');
        end
        net = load(netFilePath);
    end
    
    nViews = numel(shape);
    desc = zeros(nViews,size(feat.x,2));
    for i=1:nViews,
        if ischar(shape{i}), shape{i} = imread(shape{i}); end
        
        % get raw responses
        desc0 = get_cnn_activations(shape{i}, net, [], {feat.layerName}, ...
            'gpuMode', opts.gpuMode);
        if strcmp(feat.layerName(1:2),'fc') || ... % fully connected layers
                strcmp(feat.layerName,'prob'),
            desc0 = squeeze(desc0.(feat.layerName))';
        else
            error('feature type (%s) not yet supported.', feat.layerName);
        end
        
        % normalization & projection
        desc0 = bsxfun(@rdivide,desc0,sqrt(sum(desc0.^2,2)));
        desc0 = bsxfun(@minus,desc0,feat.pcaMean) * feat.pcaCoeff;
        desc0 = bsxfun(@rdivide,desc0,sqrt(sum(desc0.^2,2)));
        if feat.powerTrans~=1,
            desc0 = (abs(desc0).^feat.powerTrans).*sign(desc0);
        end
    end
    desc(i,:) = desc0;
    
    % retrieve by sorting distances
    switch opts.method,
        case 'mindist',
            dists = vl_alldist2(desc',refX',opts.metric);
            dists = reshape(min(reshape(dists',...
                [nRefViews nRefShapes*nViews]),[],1),[nRefShapes nViews]);
            dists = min(dists,[],2);
        case 'avgdist',
            dists = vl_alldist2(desc',refX',opts.metric);
            dists = reshape(mean(reshape(dists',...
                [nRefViews nRefShapes*nViews]),1),[nRefShapes nViews]);
            dists = mean(dists,2);
        case 'avgdesc',
            desc = mean(desc,1);
            descs = reshape(mean(reshape(refX,...
                [nRefViews nRefShapes*nDims]),1),[nRefShapes nDims]);
            dists = vl_alldist2(desc',descs',opts.metric);
        otherwise,
            error('Unknown retrieval method: %s', opts.method);
    end
    [~,I] = sort(dists,2,'ascend');
    results = refShapeIds(I(1:min(opts.nTop,nRefShapes)));
else % no query given, evaluation within dataset
    [~,I] = ismember(opts.querySets,imdb.meta.sets);
    queryImgInds = ismember(imdb.images.set,I);
    queryShapeIds=find(ismember(shapeSets,I));
    nQueryShapes = numel(queryShapeIds);
    queryX = feat.x(queryImgInds, :);
    nViews = nRefViews;
    
    % retrieve by sorting distances
    switch opts.method,
        case 'mindist',
            dists = vl_alldist2(queryX',refX',opts.metric);
            dists = -1*vl_nnpool(-1*single(dists),[nViews nRefViews], ...
                'stride', [nViews nRefViews], ...
                'method', 'max');
        case 'avgdist',
            dists = vl_alldist2(queryX',refX',opts.metric);
            dists = vl_nnpool(single(dists),[nViews nRefViews], ...
                'stride', [nViews nRefViews], ...
                'method', 'avg');
        case 'avgdesc',
            desc = reshape(mean(reshape(queryX, ...
                [nViews nQueryShapes*nDims]),1),[nQueryShapes nDims]);
            descs = reshape(mean(reshape(refX, ...
                [nRefViews nRefShapes*nDims]),1),[nRefShapes nDims]);
            dists = vl_alldist2(desc',descs',opts.metric);
        otherwise,
            error('Unknown retrieval method: %s', opts.method);
    end
    
    % pr curves 
    recall = zeros(nQueryShapes, nRefShapes+1);
    precision = zeros(nQueryShapes, nRefShapes+1);
    ap = zeros(nQueryShapes, 1);
    auc = zeros(nQueryShapes, 1);
    % interpolated pr curves
    recall_i = zeros(nQueryShapes, nRefShapes+1);
    precision_i = zeros(nQueryShapes, nRefShapes+1);
    ap_i = zeros(nQueryShapes, 1);
    auc_i = zeros(nQueryShapes, 1);
    
    for q = 1:nQueryShapes, 
        [r, p, info] = vl_pr(...
            (shapeGtClasses(refShapeIds)==shapeGtClasses(queryShapeIds(q)))-0.5, ... % LABELS
            -1*dists(q,:), ... % SCORES
            'Interpolate', false); 
        recall(q,:) = r;
        precision(q,:) = p;
        ap(q) = info.ap;
        auc(q) = info.auc;
        % interpolated
        [r, p, info] = vl_pr(...
            (shapeGtClasses(refShapeIds)==shapeGtClasses(queryShapeIds(q)))-0.5, ... % LABELS
            -1*dists(q,:), ... % SCORES
            'Interpolate', true); 
        recall_i(q,:) = r;
        precision_i(q,:) = p;
        ap_i(q) = info.ap;
        auc_i(q) = info.auc;
    end
    clear results;
    results.recall = recall;
    results.precision = precision;
    results.ap = ap;
    results.auc = auc;
    results.recall_i = recall_i;
    results.precision_i = precision_i;
    results.ap_i = ap_i;
    results.auc_i = auc_i;
end

end

