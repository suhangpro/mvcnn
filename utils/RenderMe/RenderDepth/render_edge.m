function render_edge( offfiles, readDir, saveDir, nViews )

if ~exist('nViews','var') || isempty(nViews),
    nViews = 12;
end
if ~exist('readDir', 'var') || isempty(readDir), 
    readDir = '.';
end
if ~exist('saveDir', 'var') || isempty(saveDir), 
    saveDir = '.';
end
Rs = linspace(0,2*pi,nViews+1);
Rs = Rs(1:end-1);
max_sz = 628;
im_sz = 800;
suffix = 'zedge';
nShapes = numel(offfiles);

poolobj = gcp('nocreate');
if isempty(poolobj), 
    poolSize = 0;
else
    poolSize = poolobj.NumWorkers;
end
parfor (offi = 1:nShapes, poolSize)
    fprintf(' %s\n',offfiles{offi});
    for i = 1:length(Rs),
        savePath = fullfile(saveDir,offfiles{offi});
        savePath = [savePath(1:end-4) '_' suffix '_' num2str(i) '.png']; 
        if exist(savePath,'file'), continue; end;
        
        % get depth buffer
        depth = off2im(fullfile(readDir,offfiles{offi}),3.5,Rs(i),0,2.5,0);
        depth = (depth-min(depth(:)))/(max(depth(:))-min(depth(:)));
        
        % edge map
        emap0 = edge(depth,'canny');
        region = find_object_region(emap0);
        t_sz = floor(max(region(3:4)) / max_sz * im_sz);
        [yi, xi] = ind2sub(size(emap0),find(emap0));
        yi = floor(yi + (t_sz-region(4))/2 - region(2) + 1);
        xi = floor(xi + (t_sz-region(3))/2 - region(1) + 1);
        emap = zeros(t_sz,t_sz);
        emap(sub2ind(size(emap),yi,xi)) = 1;
        emap = imdilate(emap,strel('disk',1));
        emap = imfilter(emap,fspecial('gaussian',[3 3],0.5));
        emap = max(min(imresize(emap,[im_sz im_sz]),1),0);
        emap = 1-emap;
        
        % save
        vl_xmkdir(fileparts(savePath));
        imwrite(emap, savePath);
    end
end

end



function crop = find_object_region(mask)

[ylist, xlist] = ind2sub(size(mask), find(mask));
xmin = min(xlist); xmax = max(xlist);
ymin = min(ylist); ymax = max(ylist);
crop = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1];

end
