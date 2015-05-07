function ims = render_views( mesh, varargin )
%RENDER_VIEWS render a 3d shape from multiple views 
%   mesh::
%       a mesh object containing fileds
%           .F 3 x #faces (1-based indexing)
%           .V 3 x #vertices
%       OR a path to .off file
%   `az`:: [0:30:330]
%       horizontal viewing angles
%   `el`:: 30
%       vertical elevation
%   `colorMode`:: 'gray'
%       color mode of output images (only 'gray' is supported now)
%   `outputSize`:: 224
%       output image size (both dimensions)
%   `minMargin`:: 0.1
%       minimun margin ratio in output images
%   `maxArea`:: 0.3
%       maximun area ratio in output images 
%   `figHandle`:: []
%       handle to temporary figure
%   

opts.az = [0:30:330];
opts.el = 30;
opts.colorMode = 'gray';
opts.outputSize = 224;
opts.minMargin = 0.1;
opts.maxArea = 0.3;
opts.figHandle = [];
opts = vl_argparse(opts,varargin);

if ~strcmpi(opts.colorMode,'gray'), 
    error('color mode (%s) not supported.',opts.colorMode);
end

if isempty(opts.figHandle), 
    fh = figure;
else
    fh = opts.figHandle;
end

if ischar(mesh), 
    if strcmpi(mesh(end-2:end),'off'), 
        mesh = loadMesh(mesh);
    else
        error('file type (.%s) not supported.',mesh(end-2:end));
    end
end

tmp_file = sprintf('tmp_%08s.png',dec2hex(fh));
ims = cell(1,length(opts.az));
for i=1:length(opts.az), 
    plotMesh(mesh,'solid',opts.az(i),opts.el);
    print(fh,'-dpng',tmp_file);
    im = imread(tmp_file);
    if strcmpi(opts.colorMode,'gray'), im = rgb2gray(im); end
    ims{i} = resize_im(im,opts.outputSize,opts.minMargin,opts.maxArea);
end

if isempty(opts.figHandle), close(fh); end
delete(tmp_file);

end

function im = resize_im(im,outputSize,minMargin,maxArea)

max_len = outputSize * (1-minMargin);
max_area = outputSize^2 * maxArea;

nCh = size(im,3);
mask = ~im2bw(im,1-1e-10);
mask = imfill(mask,'holes');
% blank image (all white) is outputed if not object is observed
if isempty(find(mask, 1)), 
    im = uint8(255*ones(outputSize,outputSize,nCh));
    return;
end
[ys,xs] = ind2sub(size(mask),find(mask));
y_min = min(ys); y_max = max(ys); h = y_max - y_min + 1;
x_min = min(xs); x_max = max(xs); w = x_max - x_min + 1;
scale = min(max_len/max(h,w), sqrt(max_area/sum(mask(:))));
patch = imresize(im(y_min:y_max,x_min:x_max,:),scale);
[h,w,~] = size(patch);
im = uint8(255*ones(outputSize,outputSize,nCh));
loc_start = floor((outputSize-[h w])/2);
im(loc_start(1)+(0:h-1),loc_start(2)+(0:w-1),:) = patch;

end
