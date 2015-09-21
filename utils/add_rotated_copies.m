function add_rotated_copies(imdb,saveDir,angles,flips)

if ~exist('saveDir','var') || isempty(saveDir), 
    saveDir = '.';
end
if ~exist('angles','var') || isempty(angles), 
    angles = linspace(0,360,13);
    angles = angles(1:end-1);
end
if ~exist('flips','var') || isempty(flips), 
    flips = zeros(size(angles));
end

nInstances = numel(imdb.images.name);
nViews = length(angles);

imdb2 = imdb;
imdb2.imageDir = saveDir;
imdb2.meta.nShapes = nInstances;

imdb2.images.name = cell(1,nInstances*nViews);
imdb2.images.class = reshape(repmat(imdb.images.class,[nViews 1]),[1 nInstances*nViews]);
imdb2.images.set = reshape(repmat(imdb.images.set,[nViews 1]),[1 nInstances*nViews]);
imdb2.images.id = 1:nInstances*nViews;
imdb2.images.sid = reshape(repmat(imdb.images.id,[nViews 1]),[1 nInstances*nViews]);

poolobj = gcp('nocreate');
if isempty(poolobj), 
    poolSize = 0;
else
    poolSize = poolobj.NumWorkers;
end
parfor (i=1:nInstances, poolSize)
    imPath = strrep(fullfile(imdb.imageDir,imdb.images.name{i}),'\',filesep);
    im = imread(imPath);
    fprintf(' %s\n', imdb.images.name{i});
    for ri = 1:nViews, 
        [pathstr, name, ext] = fileparts(imdb.images.name{i});
        vl_xmkdir(fullfile(saveDir,pathstr));
        savePath = fullfile(saveDir,fullfile(pathstr,[name '_' num2str(ri) ext]));
        if exist(savePath,'file'), continue; end
        im_r = 255 - imrotate(255-im,angles(ri),'crop');
        if flips(ri), im_r = fliplr(im_r); end
        imwrite(im_r,savePath);
    end
end
for i=1:nInstances, 
    for ri = 1:nViews, 
        [pathstr, name, ext] = fileparts(imdb.images.name{i});
        imdb2.images.name{(i-1)*nViews+ri} = fullfile(pathstr,[name '_' num2str(ri) ext]);
    end
    if mod(i,10)==0, fprintf('.'); end
    if mod(i,500)==0, fprintf(' [%d/%d]\n',i,nInstances); end
end
save(fullfile(saveDir,'imdb.mat'),'-struct','imdb2');

end
