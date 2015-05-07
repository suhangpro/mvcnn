function imdb_render_views( imdb, saveDir, varargin )
%IMDB_RENDER_VIEWS render 3d shapes in database
%   imdb::
%       structure containing info about all 3d shape objects
%   saveDir::
%       place to save rendered images 
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
%   `figureStartIdx`:: floor(rand()*1e8)
%       used to avoid conflict

opts.az = [0:30:330];
opts.el = 30;
opts.colorMode = 'gray';
opts.outputSize = 224;
opts.minMargin = 0.1;
opts.maxArea = 0.3;
opts.figureStartIdx = floor(rand()*1e8);
opts = vl_argparse(opts,varargin);
renderOpts = rmfield(opts,'figureStartIdx');

%{
poolObj = gcp('nocreate');
if isempty(poolObj), 
    poolSize = 1;
else
    poolSize = poolObj.NumWorkers;
end
parfor (i=1:numel(imdb.images.name),poolSize), 
%}
for i=1:numel(imdb.images.name), 
    shapePath = fullfile(imdb.imageDir,imdb.images.name{i});
    fprintf('%d %s\n',i,shapePath);
    fh = figure(i+opts.figureStartIdx-1);
    ims = render_views(shapePath,'figHandle',fh,renderOpts);
    [pathstr,namestr,extstr] = fileparts(imdb.images.name{i});
    savePathPrefix = fullfile(saveDir,pathstr);
    vl_xmkdir(savePathPrefix);
    for j = 1:numel(ims), 
        savePath = fullfile(savePathPrefix,sprintf('%s_%03d.png',namestr,j));
        imwrite(ims{j},savePath);
    end
    close(fh);
end

end

