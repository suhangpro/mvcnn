function [ results,info ] = retrieve_shapes_cnn( shape, feat, varargin )
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
%       other choices: 'maxdesc','avgdist','mindist','avgmindist','minavgdist'
%   `net`::[]
%       preloaded cnn model (required when feat.modelName is not available)
%   `gpuMode`:: false
%       set to true to compute on GPU
%   `numWorkers`:: 1
%       number of workers used to compute pairwise 
%   `nTop`:: Inf
%       number of images in results
%   `querySets`:: {'test'}
%       set of query shapes (used only when shape==[])
%   `refSets`:: {'test'}
%       set of reference images
%   `metric`:: 'L2'
%       other choices: 'LINF', 'L1', 'L0', 'CHI2', 'HELL'
%   `multiview`:: true
%       set to false to evaluate on single views of each instance 
%   `logPath:: []
%       place to save log information
%
% Hang Su

% default options
opts.method = 'avgdesc';
opts.net = [];
opts.gpuMode = false;
opts.numWorkers = 1;
opts.nTop = Inf;
opts.querySets = {'test'};
opts.refSets = {'test'};
opts.metric = 'L2';
opts.multiview = true;
opts.logPath = [];
[opts, varargin] = vl_argparse(opts,varargin);

if opts.numWorkers>1, 
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers<opts.numWorkers, 
        if ~isempty(pool), delete(pool); end
        pool = parpool(opts.numWorkers);
    end
end

if isequal(opts.querySets,opts.refSets), 
    isSelfRef = true;
else
    isSelfRef = false;
end

% -------------------------------------------------------------------------
%                       sort imdb.images & feat.x w.r.t (sid,view) or (id)
% -------------------------------------------------------------------------
imdb = feat.imdb;

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
if isfield(imdb.images,'sid'), 
    shapeStartIdx = find(imdb.images.sid-circshift(imdb.images.sid,[0 1]));
    shapeIds = imdb.images.sid(shapeStartIdx);
else
    shapeStartIdx = 1:length(imdb.images.id);
    shapeIds = imdb.images.id;
end
shapeGtClasses = imdb.images.class(shapeStartIdx);
shapeSets = imdb.images.set(shapeStartIdx);

% refX
[~,I] = ismember(opts.refSets,imdb.meta.sets);
shapeRefInds = ismember(shapeSets,I);
refShapeIds = shapeIds(shapeRefInds);
nRefShapes = numel(refShapeIds);
if ~pooledFeat && isfield(imdb.images,'sid'), 
    nRefDescPerShape = sum(imdb.images.sid==refShapeIds(1));
else
    nRefDescPerShape = 1;
end
if isfield(imdb.images,'sid'), 
    if pooledFeat, 
        descRefInds = shapeRefInds;
    else
        descRefInds = ismember(imdb.images.sid,refShapeIds); 
    end
else
    descRefInds = ismember(imdb.images.id,refShapeIds); 
end
refX = feat.x(descRefInds, :);
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
    [~,I] = ismember(opts.querySets,imdb.meta.sets);
    shapeQueryInds = ismember(shapeSets,I);
    queryShapeIds = shapeIds(shapeQueryInds);
    nQueryShapes = numel(queryShapeIds);
    if ~pooledFeat && isfield(imdb.images,'sid'), 
        nDescPerShape = sum(imdb.images.sid==queryShapeIds(1));
    else
        nDescPerShape = 1;
    end
    if isfield(imdb.images,'sid'), 
        if pooledFeat, 
            descQueryInds = shapeQueryInds;
        else
            descQueryInds = ismember(imdb.images.sid,queryShapeIds); 
        end
    else
        descQueryInds = ismember(imdb.images.id,queryShapeIds); 
    end
    queryX = feat.x(descQueryInds, :);
    
    nDescPerShapeOri = nDescPerShape;
    nQueryShapesOri = nQueryShapes;
    if ~opts.multiview,
        nDescPerShape = 1;
        nQueryShapes = nQueryShapes * nDescPerShapeOri;
    end
    
end

% -------------------------------------------------------------------------
%                                            retrieve by sorting distances
% -------------------------------------------------------------------------
switch opts.method,
    case 'mindist',
        dists0 = par_alldist(queryX',refX',...
            'measure',opts.metric,'numWorkers',opts.numWorkers);
        dists = -1*vl_nnpool(-1*single(dists0),[nDescPerShape nRefDescPerShape], ...
            'stride', [nDescPerShape nRefDescPerShape], ...
            'method', 'max');
    case 'avgdist',
        dists0 = par_alldist(queryX',refX',...
            'measure',opts.metric,'numWorkers',opts.numWorkers);
        dists = vl_nnpool(single(dists0),[nDescPerShape nRefDescPerShape], ...
            'stride', [nDescPerShape nRefDescPerShape], ...
            'method', 'avg');
    case 'avgmindist',
        dists0 = par_alldist(queryX',refX',...
            'measure',opts.metric,'numWorkers',opts.numWorkers);
        dists_1 = -1*vl_nnpool(-1*single(dists0),[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','max');
        dists_1 = vl_nnpool(dists_1,[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','avg');
        dists_2 = -1*vl_nnpool(-1*single(dists0),[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','max');
        dists_2 = vl_nnpool(dists_2,[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','avg');
        dists = 0.5*(dists_1+dists_2);
    case 'minavgdist',
        dists0 = par_alldist(queryX',refX',...
            'measure',opts.metric,'numWorkers',opts.numWorkers);
        dists_1 = vl_nnpool(single(dists0),[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','avg');
        dists_1 = -1*vl_nnpool(-1*dists_1,[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','max');
        dists_2 = vl_nnpool(single(dists0),[1 nRefDescPerShape], ...
            'stride',[1 nRefDescPerShape], 'method','avg');
        dists_2 = -1*vl_nnpool(-1*dists_2,[nDescPerShape 1], ...
            'stride',[nDescPerShape 1], 'method','max');
        dists = 0.5*(dists_1+dists_2);
    case 'avgdesc',
        desc = reshape(mean(reshape(queryX, ...
            [nDescPerShape nQueryShapes*nDims]),1),[nQueryShapes nDims]);
        descs = reshape(mean(reshape(refX, ...
            [nRefDescPerShape nRefShapes*nDims]),1),[nRefShapes nDims]);
        dists0 = par_alldist(desc',descs',...
            'measure',opts.metric,'numWorkers',opts.numWorkers);
        dists = dists0;
    case 'maxdesc',
        desc = reshape(max(reshape(queryX, ...
            [nDescPerShape nQueryShapes*nDims]),[],1),[nQueryShapes nDims]);
        descs = reshape(max(reshape(refX, ...
            [nRefDescPerShape nRefShapes*nDims]),[],1),[nRefShapes nDims]);
        dists0 = par_alldist(desc',descs',...
            'measure',opts.metric,'numWorkers',opts.numWorkers);
        dists = dists0;
    otherwise,
        error('Unknown retrieval method: %s', opts.method);
end

% -------------------------------------------------------------------------
%                                               prepare and return results
% -------------------------------------------------------------------------
if ~isempty(shape), % retrieval given a query shape
    [~,I] = sort(dists,'ascend');
    results = refShapeIds(I(1:min(opts.nTop,nRefShapes)));
    info = [];
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
    
    refClass = shapeGtClasses(refShapeIds);
    refClass = repmat(refClass, [nQueryShapes, 1]);
    queryClass = shapeGtClasses(queryShapeIds);
    if ~opts.multiview, 
        queryClass = reshape(repmat(queryClass,[nDescPerShapeOri 1]), ...
            [1 nQueryShapes]);
    end
    qrDists = dists;
    
    if isSelfRef, 
        assert(nRefShapes==nQueryShapesOri);
        recall(:,end) = [];
        precision(:,end) = [];
        recall_i(:,end) = [];
        precision_i(:,end) = [];
        refClass = rmDiagOnRow(refClass);
        qrDists = rmDiagOnRow(qrDists);
    end
    if opts.numWorkers>1,
        parfor q = 1:nQueryShapes,
            [r, p, info] = vl_pr(...
                (refClass(q,:)==queryClass(q))-0.5, ... % LABELS
                -1*qrDists(q,:), ... % SCORES
                'Interpolate', false);
            recall(q,:) = r;
            precision(q,:) = p;
            ap(q) = info.ap;
            auc(q) = info.auc;
            % interpolated
            [r, p, info] = vl_pr(...
                (refClass(q,:)==queryClass(q))-0.5, ... % LABELS
                -1*qrDists(q,:), ... % SCORES
                'Interpolate', true);
            recall_i(q,:) = r;
            precision_i(q,:) = p;
            ap_i(q) = info.ap;
            auc_i(q) = info.auc;
        end
    else
        for q = 1:nQueryShapes,
            [r, p, info] = vl_pr(...
                (refClass(q,:)==queryClass(q))-0.5, ... % LABELS
                -1*qrDists(q,:), ... % SCORES
                'Interpolate', false);
            recall(q,:) = r;
            precision(q,:) = p;
            ap(q) = info.ap;
            auc(q) = info.auc;
            % interpolated
            [r, p, info] = vl_pr(...
                (refClass(q,:)==queryClass(q))-0.5, ... % LABELS
                -1*qrDists(q,:), ... % SCORES
                'Interpolate', true);
            recall_i(q,:) = r;
            precision_i(q,:) = p;
            ap_i(q) = info.ap;
            auc_i(q) = info.auc;
        end
    end
    info = [];
    info.recall = recall;
    info.precision = precision;
    info.ap = ap;
    info.auc = auc;
    info.recall_i = recall_i;
    info.precision_i = precision_i;
    info.ap_i = ap_i;
    info.auc_i = auc_i;
    clear results;
    results.dists = dists; 
    results.dists0 = dists0;
    [~,I] = sort(dists,2,'ascend');
    results.rankings = refShapeIds(I(:,1:min(opts.nTop,nRefShapes)));
    
    % output to log
    if ~isempty(opts.logPath), 
        fprintf('Evaluation finished! \n');
        fprintf('\tdataset: %s\n', imdb.imageDir);
        fprintf('\tmodel: %s\n',feat.modelName);
        fprintf('\tlayer: %s\n',feat.layerName);
        fprintf('\tmethod: %s\n', opts.method);
        fprintf('\tmAP: %g%%\n',mean(info.ap)*100);
        fprintf('\tAUC: %g%%\n',mean(info.auc)*100);
        fprintf('\tmAP (interpolated): %g%%\n',mean(info.ap_i)*100);
        fprintf('\tAUC (interpolated): %g%%\n',mean(info.auc_i)*100);
        
        fid = fopen(opts.logPath,'a+');
        fprintf(fid, '(%s) -- Retrieval\n', datestr(now));
        fprintf(fid, '\tdataset: %s\n', imdb.imageDir);
        fprintf(fid, '\tmodel: %s\n',feat.modelName);
        fprintf(fid, '\tlayer: %s\n',feat.layerName);
        fprintf(fid, '\tmethod: %s\n', opts.method);
        fprintf(fid, '\tmAP: %g%%\n',mean(info.ap)*100);
        fprintf(fid, '\tAUC: %g%%\n',mean(info.auc)*100);
        fprintf(fid, '\tmAP (interpolated): %g%%\n',mean(info.ap_i)*100);
        fprintf(fid, '\tAUC (interpolated): %g%%\n',mean(info.auc_i)*100);
        fclose(fid);
    end
    
end

end

function M = rmDiagOnRow(M)
    assert(ismatrix(M));
    [sz1, sz2] = size(M);
    if mod(sz1,sz2)==0, 
        sy = sz1/sz2;
        sx = 1;
    elseif mod(sz2,sz1)==0, 
        sy = 1;
        sx = sz2/sz1;
    elseif sz1==sz2,
        u = triu(M,1);
        l = tril(M,-1);
        M = u(:,2:end) + l(:,1:end-1);
        return; 
    else
        error('Wrong matrix size: [%d %d]',sz1, sz2);
    end
    ix = 1;
    M0 = M;
    M = zeros(sz1, sz2-sx);
    for iy=1:sy:sz1, 
        M(iy:iy+sy-1,:) = M0(iy:iy+sy-1,[1:ix-1 ix+sx:sz2]);
        ix = ix + sx;
    end
end
