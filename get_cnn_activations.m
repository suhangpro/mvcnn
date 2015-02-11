function [ feat ] = get_cnn_activations( im, net, subWins, layers, varargin)
%GET_CNN_FEATURE Get CNN activation responses for im
%   im:: 
%       image matrix
%   net::
%       cnn model structure 
%   subWins:: [0; 1; 0; 1; 0]
%       see get_augmentation_matrix.m for details 
%   layers:: {'fc6'}
%       can be either a structure (.name, .sizes, .index) or string array 
%   `gpuMode`:: false 
%       set to true to compute on GPU 

if nargin<4 || isempty(layers), 
    layers = {'fc6'};
end
if nargin<3 || isempty(subWins), 
    subWins = get_augmentation_matrix('none');
end
nSubWins = size(subWins,2);

opts.gpuMode = false;
opts = vl_argparse(opts,varargin);

if iscell(layers),
    layersName = layers;
    layers = struct;
    % name 
    layers.name = layersName;
    % index 
    allLayersName = cell(1,length(net.layers));
    for i=1:length(allLayersName),
        allLayersName{i} = net.layers{i}.name;
    end
    [~,layers.index] = ismember(layers.name,allLayersName);
    layers.index = layers.index + 1;
    % sizes
    im0 = single(imread('peppers.png'));
    im0_ = imresize(im0,net.normalization.imageSize(1:2));
    if opts.gpuMode, im0_ = gpuArray(im0_); end
    res = vl_simplenn(net,im0_);
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
if size(im,3)==1, im = repmat(im,[1,1,3]); end;

for ri = 1:nSubWins,
    r = subWins(1:4,ri).*[size(im,2);size(im,2);size(im,1);size(im,1)];
    r = round(r);
    im_ = im(max(1,r(3)):min(size(im,1),r(3)+r(4)),...
        max(1,r(1)):min(size(im,2),r(1)+r(2)),:);
    if subWins(5,ri), im_ = flipdim(im_,2); end
    im_ = imresize(im_, net.normalization.imageSize(1:2))...
        - net.normalization.averageImage;
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

