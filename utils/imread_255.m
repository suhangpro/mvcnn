function im = imread_255( path, nChannels )
%IMREAD_255 A wrapper of imread that always returns image of nChannels
%channels and with single precision values in range [0.0,255.0]
%   

if nargin==1, nChannels = 3; end 

if ~(nChannels==1 || nChannels==3), 
    error('Unsupported type of image format: %d channels', nChannels);
end

im = imread(strrep(path,'\',filesep)); 

if size(im,3)~=nChannels, 
    if size(im,3)==3, 
        im = rgb2gray(im); 
    elseif size(im,3)==1, 
        im = repmat(im,[1,1,3]); 
    else
        error('Unsupported image format: %d channels', size(im,3));
    end
end

if ~isa(im,'uint8'), 
    im = im2single(im) * 255; 
else
    im = single(im); 
end

end

