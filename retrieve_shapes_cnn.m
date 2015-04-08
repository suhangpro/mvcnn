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
%   `method`:: 'avgdesc'
%       other choices: 'avgdist', 'mindist', 'avgmindist', 'minavgdist'
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
opts.method = 'avgdesc';
opts.net = [];
opts.gpuMode = false;
opts.nTop = Inf;
opts.querySets = {'test'};
opts.refSets = {'test'};
opts.metric = 'L2';
opts = vl_argparse(opts,varargin);

% -------------------------------------------------------------------------
%                       sort imdb.images & feat.x w.r.t (sid,view) or (id)
% -------------------------------------------------------------------------
imdb = feat.imdb;
if ~isfield(imdb.meta,'nShapes'), 
    imdb.meta.nShapes = numel(imdb.images.name); 
end;

% sort imdb.images wrt id
[imdb.images.id, I] = sort(imdb.images.id);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);
if isfield(imdb.images,'sid'), imdb.images.sid = imdb.images.sid(I); end

% sort feat.x wrt id/sid
if isfield(feat, 'sid'), 
    pooledFeat = true; 
    [feat.sid,I] = sort(feat.sid);
    feat.x = feat.x(I,:);
else
    pooledFeat = false; 
    [feat.id,I] = sort(feat.id);
    feat.x = feat.x(I,:);
end

% sort imdb.images wrt sid
if isfield(imdb.images,'sid'),
    [imdb.images.sid, I] = sort(imdb.images.sid);
    imdb.images.name = imdb.images.name(I);
    imdb.images.class = imdb.images.class(I);
    imdb.images.set = imdb.images.set(I);
    imdb.images.id = imdb.images.id(I);
    if ~pooledFeat, feat.x = feat.x(I,:); end
end

% -------------------------------------------------------------------------
%                                                      feature descriptors
% -------------------------------------------------------------------------
nRefViews = length(imdb.images.name)/imdb.meta.nShapes;
nRefDescPerShape = size(feat.x,1)/imdb.meta.nShapes;
shapeGtClasses = imdb.images.class(1:nRefViews:end);
shapeSets = imdb.images.set(1:nRefViews:end);

% refX
[~,I] = ismember(opts.refSets,imdb.meta.sets);
refShapeIds=find(ismember(shapeSets,I));
nRefShapes = numel(refShapeIds);
tmp = zeros(nRefDescPerShape, imdb.meta.nShapes);
tmp(:,refShapeIds) = 1;
refX = feat.x(find(tmp)', :);
nDims = size(refX,2);

% queryX
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
	% get raw responses
    nViews = numel(shape);
    for i=1:numel(shape), 
        if ischar(shape{i}), shape{i} = imread(shape{i}); end
    end
    if pooledFeat, 
        queryX = get_cnn_activations(shape, net, [], {feat.layerName}, ...
            'gpuMode', opts.gpuMode);
        queryX = squeeze(queryX.(feat.layerName))';
        nDescPerShape = size(queryX,1);
    else
        nDescPerShape = nViews;
        queryX = zeros(nDescPerShape, nDims);
        for i=1:nDescPerShape,
            desc0 = get_cnn_activations(shape{i}, net, [], {feat.layerName}, ...
                'gpuMode', opts.gpuMode);
            desc0 = squeeze(desc0.(feat.layerName))';
        end
        queryX(i,:) = desc0;
    end    
    % normalization & projection
    if isfield(feat,'pcaMean'),
        queryX = bsxfun(@rdivide,queryX,sqrt(sum(queryX.^2,2)));
        queryX = bsxfun(@minus,queryX,feat.pcaMean) * feat.pcaCoeff;
        queryX = bsxfun(@rdivide,queryX,sqrt(sum(queryX.^2,2)));
        if feat.powerTrans~=1,
            queryX = (abs(queryX).^feat.powerTrans).*sign(queryX);
        end
    end
else 
    nDescPerShape = nRefDescPerShape;
    [~,I] = ismember(opts.querySets,imdb.meta.sets);
    queryShapeIds=find(ismember(shapeSets,I));
    nQueryShapes = numel(queryShapeIds);
    tmp = zeros(nDescPerShape, imdb.meta.nShapes);
    tmp(:,queryShapeIds) = 1;
    queryX = feat.x(find(tmp)', :);
end

% -------------------------------------------------------------------------
%                                            retrieve by sorting distances
% -------------------------------------------------------------------------
switch opts.method,
    case 'mindist',
        dists = vl_alldist2(queryX',refX',opts.metric);
        dists = -1*vl_nnpool(-1*single(dists),[nDescPerShape nRefDescPerShape], ...
            'stride', [nDescPerShape nRefDescPerShape], ...
            'method', 'max');
    case 'avgdist',
        dists = vl_alldist2(queryX',refX',opts.metric);
        dists = vl_nnpool(single(dists),[nDescPerShape nRefDescPerShape], ...
            'stride', [nDescPerShape nRefDescPerShape], ...
            'method', 'avg');
    case 'avgmindist',
        dists = vl_alldist2(queryX',refX',opts.metric);
        dists_1 = -1*vl_nnpool(-1*single(dists),[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','max');
        dists_1 = vl_nnpool(dists_1,[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','avg');
        dists_2 = -1*vl_nnpool(-1*single(dists),[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','max');
        dists_2 = vl_nnpool(dists_2,[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','avg');
        dists = 0.5*(dists_1+dists_2);
    case 'minavgdist',
        dists = vl_alldist2(queryX',refX',opts.metric);
        dists_1 = vl_nnpool(single(dists),[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','avg');
        dists_1 = -1*vl_nnpool(-1*dists_1,[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','max');
        dists_2 = vl_nnpool(single(dists),[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','avg');
        dists_2 = -1*vl_nnpool(-1*dists_2,[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','max');
        dists = 0.5*(dists_1+dists_2);
    case 'avgdesc',
        desc = reshape(mean(reshape(queryX, ...
            [nDescPerShape nQueryShapes*nDims]),1),[nQueryShapes nDims]);
        descs = reshape(mean(reshape(refX, ...
            [nRefDescPerShape nRefShapes*nDims]),1),[nRefShapes nDims]);
        dists = vl_alldist2(desc',descs',opts.metric);
    otherwise,
        error('Unknown retrieval method: %s', opts.method);
end

% -------------------------------------------------------------------------
%                                               prepare and return results
% -------------------------------------------------------------------------
if ~isempty(shape), % retrieval given a query shape
    [~,I] = sort(dists,'ascend');
    results = refShapeIds(I(1:min(opts.nTop,nRefShapes)));
else                % no query given, evaluation within dataset
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

