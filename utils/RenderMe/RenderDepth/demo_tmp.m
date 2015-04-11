function demo_tmp()
% Library:
% We use the Off-screen Rendering library from Mesa3D:
% http://www.mesa3d.org/osmesa.html
% You will need to have libosmesa6-dev or newer version of osmesa installed.
% in linux command line: sudo apt-get install libosmesa6-dev

% install lOSMesa
% compile
% mex RenderMex.cpp -lGLU -lOSMesa
% or
% mex RenderMex.cpp -lGLU -lOSMesa -I/media/Data/usr/Mesa-9.1.2/include

% offfile = 'chair000009.off';

offfile_dir = 'data/modelnet40off/testing/bathtub';
toonedge_dir = 'data/modelnet40toonedge/testing/bathtub';
shape_name = 'bathtub_000000155';
offfile = fullfile(offfile_dir,[shape_name '.off']);

rs = 0:pi/6:pi*3/6;
for ri = 1:length(rs),  

depth = off2im(offfile,3.5,rs(ri),0,2.5,0);
depth = (depth-min(depth(:)))/(max(depth(:))-min(depth(:)));

max_sz = 628;
im_sz = 800;

emap0 = edge(depth,'canny');
region = find_object_region(emap0);
t_sz = floor(max(region(3:4)) / max_sz * im_sz);

[yi, xi] = ind2sub(size(emap0),find(emap0));
yi = floor(yi + (t_sz-region(4))/2 - region(2) + 1);
xi = floor(xi + (t_sz-region(3))/2 - region(1) + 1);
emap = zeros(t_sz,t_sz);
emap(sub2ind(size(emap),yi,xi)) = 1;
emap = max(min(imresize(emap,[im_sz im_sz]),1),0);
emap = imdilate(emap,strel('disk',1));
emap = imfilter(emap,fspecial('gaussian',[3 3],0.5));
emap = 1-emap;
vl_tightsubplot(2,4,ri);
imshow(emap);

im = imread(fullfile(toonedge_dir,[shape_name '_toonedge_' num2str(ri) '.png']));
vl_tightsubplot(2,4,ri+4);
imshow(im);

end

end

function crop = find_object_region(mask)

[ylist, xlist] = ind2sub(size(mask), find(mask));
xmin = min(xlist); xmax = max(xlist);
ymin = min(ylist); ymax = max(ylist);
crop = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1];

end
