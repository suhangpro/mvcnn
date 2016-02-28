function [ feat ] = cnn_shape_get_activations(im, net, layers, subWins, varargin)
%CNN_SHAPE_GET_ACTIVATIONS Get CNN activation responses for im
%   im:: 
%       image matrix, #channels (size(im,3)) must be compatible with net
%       0~255
%   net::
%       cnn model structure 
%   layers:: {'relu7'}
%       can be either a structure (.name, .sizes, .index) or string array 
%   subWins:: [0; 1; 0; 1; 0]
%       see get_augmentation_matrix.m for details 
%   `gpuMode`:: false 
%       set to true to compute on GPU 

if ~exist('layers','var') || isempty(layers), 
    layers = {'relu7'};
end
if ~exist('subWins','var') || isempty(subWins), 
    subWins = get_augmentation_matrix('none');
end

opts.gpuMode = false;
opts = vl_argparse(opts,varargin);

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

% find if net contains viewpool layer
viewpoolIdx = find(cellfun(@(x)strcmp(x.name, 'viewpool'),net.layers));
if ~isempty(viewpoolIdx), 
    if numel(viewpoolIdx)>1, 
        error('More than one viewpool layers found!'); 
    end
    nViews = net.layers{viewpoolIdx}.vstride;
else
    nViews = 1;
end

if iscell(layers),
    layersName = layers;
    layers = struct;
    % name 
    layers.name = layersName;
    % index 
    allLayersName = cellfun(@(s) s.name,net.layers,'UniformOutput',false); 
    [~,layers.index] = ismember(layers.name,allLayersName);
    layers.index = layers.index + 1;
    % sizes
    im0 = zeros(net.meta.normalization.imageSize(1), ...
        net.meta.normalization.imageSize(2), nChannels, nViews, 'single') * 255;
    if opts.gpuMode, im0 = gpuArray(im0); end
    res = vl_simplenn(net,im0);
    layers.sizes = zeros(3,numel(layers.name));
    for i = 1:numel(layers.name),
        layers.sizes(:,i) = reshape(size(res(layers.index(i)).x),[3,1]);
    end
end

featCell = cell(1,numel(layers.name));
for fi = 1:numel(layers.name),
    featCell{fi} = zeros(layers.sizes(1,fi),layers.sizes(2,fi),...
        layers.sizes(3,fi),nSubWins);
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
    if opts.gpuMode,
        im_ = gpuArray(im_);
    end
    res = vl_simplenn(net,im_);
    for fi = 1:numel(layers.name),
        featCell{fi}(:,:,:,ri) = gather(res(layers.index(fi)).x);
    end
end
% pack features into structure
feat = struct;
for fi = 1:numel(layers.name),
    feat.(layers.name{fi}) = featCell{fi};
end

end

