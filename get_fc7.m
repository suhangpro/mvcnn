function [ feat ] = get_fc7( net, imdb, useGpu )
%UNTITLED2 Summary of this function goes here
%   feat 4096 x #images

if exist('useGpu','var') && useGpu, 
    net = vl_simplenn_move(net,'gpu');
else
    useGpu = false;
    net = vl_simplenn_move(net,'cpu');
end

feat = zeros(4096,length(imdb.images.name));

for i = 1:length(imdb.images.name), 
    im = imread(fullfile(imdb.imageDir,imdb.images.name{i}));
    im_ = single(im);
    if size(im_,3)==1, im_ = repmat(im_,[1,1,3]); end
    im_ = imresize(im_,net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    if useGpu, 
        im_ = gpuArray(im_);
    end        
    res = vl_simplenn(net,im_);
    feat(:,i) = squeeze(gather(res(20).x));
    if mod(i,100)==0, fprintf('.'); end
    if mod(i,1000)==0, fprintf(' %d/%d\n',i,length(imdb.images.name)); end

end
    

end


