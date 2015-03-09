function [ results ] = retrieve_images_cnn( im, feat, varargin )
%RETRIEVE_IMAGES_CNN Retrieve images using CNN features 
% 
%   im:: 
%       image or path to it
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
%   `net`::[]
%       preloaded cnn model (required when feat.modelName is not available)
%   `augmentation`:: 'fr2'
%       1st field(f|n) indicates whether include flipped copy or not
%       2nd field(s|r) indicates type of region - Square or Rectangle
%       3rd field(1..4) indicates number of levels 
%       note: 'none', 'ns1', 'nr1' are equivalent
%   `gpuMode`:: false
%       set to true to compute on GPU 
%   `nTop`:: Inf
%       number of images in results 
%   `metric`:: 'L2'
%       other choices: 'LINF', 'L1', 'L0', 'CHI2', 'HELL'

% default options
opts.net = [];
opts.augmentation = 'fr2';
opts.gpuMode = false;
opts.nTop = Inf;
opts.metric = 'L2';
opts = vl_argparse(opts,varargin);

% data augmentation
subWins = get_augmentation_matrix(opts.augmentation);

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

if ischar(im), im = imread(im); end

% get raw responses 
desc = get_cnn_activations(im, net, subWins, {feat.layerName}, ...
    'gpuMode', opts.gpuMode); 
if strcmp(feat.layerName(1:2),'fc'), % fully connected layers 
    desc = squeeze(desc.(feat.layerName))';
else
    error('feature type (%s) not yet supported.', feat.layerName);
end

% normalization & projection
desc = bsxfun(@rdivide,desc,sqrt(sum(desc.^2,2)));
desc = bsxfun(@minus,desc,feat.pcaMean) * feat.pcaCoeff;
desc = bsxfun(@rdivide,desc,sqrt(sum(desc.^2,2)));
if feat.powerTrans~=1, 
    desc = (abs(desc).^feat.powerTrans).*sign(desc);
end

% retrieve by sorting distances 
dists = vl_alldist2(desc',feat.x',opts.metric); 
nRefImg = numel(feat.imdb.images.name);
nRefSW = length(feat.id)/nRefImg;
nSW = size(subWins,2);
dists = reshape(min(reshape(dists',[nRefSW nRefImg*nSW]),[],1),[nRefImg nSW])'; 
dists = mean(dists,1);
[~,Idxs] = sort(dists,2,'ascend');
Idxs = Idxs(1:min(opts.nTop,nRefImg));
results = cellfun(@(s) fullfile(feat.imdb.imageDir,s), ...
    feat.imdb.images.name(Idxs), 'UniformOutput', false);

end

