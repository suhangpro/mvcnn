function depth = off2im(offfile, ratio, xzRot, tilt, objz, objx)

if ~exist('xzRot', 'var')
    xzRot = rand() * pi*2;
end
if ~exist('tilt', 'var')
    tilt = - rand() * 0.1 * pi;
end
if ~exist('objz', 'var')
    objz = 1.5 + rand() * 4;   % 1.5 to 5.5
end
if ~exist('objx', 'var')
    objx = (rand()*0.5-0.25) .* objz;
end
if ~exist('ratio', 'var')
    ratio = 2;
end

%% Camera Paramter
fx_rgb = 5.1885790117450188e+02 * ratio;
fy_rgb = 5.1946961112127485e+02 * ratio;
cx_rgb = 3.2558244941119034e+02 * ratio;
cy_rgb = 2.5373616633400465e+02 * ratio;
K=[fx_rgb 0 cx_rgb; 0 fy_rgb cy_rgb; 0 0 1];
imw = 600 * ratio; 
imh = 600 * ratio;
C = [0;1.7;0];    % y off should be 1.7
z_near = 0.3;
z_far_ratio = 1.2;
Ryzswi = [1, 0, 0; 0, 0, 1; 0, 1, 0];

%%
offobj = offLoader(offfile);
offobj.vmat = Ryzswi * offobj.vmat;
Robj = genRotMat(xzRot);
Rcam = genTiltMat(tilt);
P = K * Rcam * [eye(3), -C];
vmat = scalePoints(Robj * offobj.vmat, [objx;1.3;objz], [1;1;1]);
result = RenderMex(P, imw, imh, vmat, uint32(offobj.fmat))';
depth = z_near./(1-double(result)/2^32);
if isempty(find(abs(depth)<100)), 
    depth = [];
    return;
end
maxDepth = max(depth(abs(depth) < 100));
cropmask = (depth < z_near) | (depth > z_far_ratio * maxDepth);
crop = findCropRegion(~cropmask);
depth = depth(crop(1)+(0:crop(3)-1), crop(2)+(0:crop(4)-1));
depth(cropmask(crop(1)+(0:crop(3)-1), crop(2)+(0:crop(4)-1))) = z_far_ratio * maxDepth;

end

function crop = findCropRegion(mask)

[xlist, ylist] = ind2sub(size(mask), find(mask));
xmin = min(xlist); xmax = max(xlist);
ymin = min(ylist); ymax = max(ylist);
crop = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1];

margin_ratio = 0.2;
pad = floor(margin_ratio*crop(3:4));
crop = [crop(1)-pad(1), ...
    crop(2)-pad(2), ...
    crop(3)+pad(1)*2, ...
    crop(4)+pad(2)*2, ...
    ];
crop = max(crop,1);
crop(3:4) = min(crop(3:4),size(mask)-crop(1:2)+1);

end

function R = genRotMat(theta)

R = [cos(theta), 0, -sin(theta)
    0, 1, 0
    sin(theta), 0, cos(theta)];

end

function R = genTiltMat(theta)

R = [1, 0, 0
    0, cos(theta), -sin(theta)
    0,  sin(theta), cos(theta)];

end

function coornew = scalePoints(coor, center, size)

% function coornew = scalePoints(coor, box)
% 
% parameters:
%   coor: 3*n coordinates of n points
%   center: 3*1 the center of new point cloud
%   size: 3*1 the size of new point cloud

minv = min(coor, [], 2);
maxv = max(coor, [], 2);
oldCenter = (minv+maxv)/2;
oldSize = maxv - minv;
scale = min(size ./ oldSize);
coornew = bsxfun(@plus, scale * coor, center-scale*oldCenter);

end
