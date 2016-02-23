function imdb = setup_imdb_modelnet(datasetDir, varargin)

opts.seed = 0 ;             % random seed generator
opts.ratio = [0.8 0.2];     % train:val ratio
opts.ext = '.jpg';          % extension of target files
opts.extmesh = '.off';      % extension of target mesh files
opts.useUprightAssumption = true; % if true, 12 views will be used to render meshes, otherwise 80 views based on a dodecahedron
opts = vl_argparse(opts, varargin);

opts.ratio = opts.ratio(1:2)/sum(opts.ratio(1:2));
rng(opts.seed);
imdb.imageDir = datasetDir;

% meta
folders = {};
fprintf('Scanning for classes ... ');
contents = dir(imdb.imageDir);
for i=1:numel(contents),
    if contents(i).isdir, folders = [folders contents(i).name]; end
end
imdb.meta.classes = setdiff(folders,{'.','..'});
imdb.meta.sets = {'train', 'val', 'test'};
fprintf('%d classes found! \n', length(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.class   = [];
imdb.images.set     = [];
imdb.images.sid     = [];
fprintf('Scanning for images: \n');
[imdb, nTrainShapes] = scan_images(imdb, opts);
if nTrainShapes == 0
    fprintf('No images found. Attempting to scan for meshes and render them...\n')
    numShapes = render_meshes(imdb, opts);
    if numShapes == 0
        error('No image or shape data found!');
    else
        imdb = scan_images(imdb, opts);
    end
end

[imdb.images.sid, I] = sort(imdb.images.sid);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);

nViews = sum(imdb.images.sid==1); 
assert((nViews==12 && opts.useUprightAssumption) ...
    || (nViews==80 && ~opts.useUprightAssumption), ...
    'Number of views (%d) wrong', nViews); 
imdb.images.id = 1:length(imdb.images.name);

end

function [imdb, nTrainShapes] = scan_images(imdb, opts)
for ci = 1:length(imdb.meta.classes),
    fprintf('  [%2d/%2d] %s ... ', ci, length(imdb.meta.classes), ...
        imdb.meta.classes{ci});
    trainDir = fullfile(imdb.imageDir,imdb.meta.classes{ci},'train');
    valDir = fullfile(imdb.imageDir,imdb.meta.classes{ci},'val');
    testDir = fullfile(imdb.imageDir,imdb.meta.classes{ci},'test');
    if exist(trainDir,'dir'), hasTrain = true; else hasTrain = false; end;
    if exist(valDir,'dir'), hasVal = true; else hasVal = false; end;
    if exist(testDir,'dir'), hasTest = true; else hasTest = false; end;
    
    % train
    if hasTrain,
        files = dir(fullfile(trainDir, ['*' opts.ext]));
        fileNames = {files.name};
        nTrainImages = length(fileNames);
        imVids = cellfun(@(s) get_shape_vid(s), fileNames);
        [~,I] = sort(imVids);
        fileNames = fileNames(I); % order images wrt view id
        sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
            'UniformOutput', false);
        sNamesUniq = unique(sNames);
        nTrainShapes = length(sNamesUniq);
        [~,imSids] = ismember(sNames,sNamesUniq);
        if isempty(imdb.images.sid), maxSid = 0;
        else maxSid = max(imdb.images.sid); end
        imdb.images.sid = [imdb.images.sid maxSid+imSids];
        imdb.images.set = [imdb.images.set ones(1,nTrainImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nTrainImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},'train',s), ...
            fileNames, 'UniformOutput',false)];
    else
        nTrainShapes = 0;
    end
    
    % val
    if hasVal,
        files = dir(fullfile(valDir, ['*' opts.ext]));
        fileNames = {files.name};
        nValImages = length(fileNames);
        imVids = cellfun(@(s) get_shape_vid(s), fileNames);
        [~,I] = sort(imVids);
        fileNames = fileNames(I); % order images wrt view id
        sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
            'UniformOutput', false);
        sNamesUniq = unique(sNames);
        nValShapes = length(sNamesUniq);
        [~,imSids] = ismember(sNames,sNamesUniq);
        if isempty(imdb.images.sid), maxSid = 0;
        else maxSid = max(imdb.images.sid); end
        imdb.images.sid = [imdb.images.sid maxSid+imSids];
        imdb.images.set = [imdb.images.set 2*ones(1,nValImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nValImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},'val',s), ...
            fileNames, 'UniformOutput',false)];
    elseif hasTrain && opts.ratio(2)>0,
        inds = (imdb.images.set==1 & imdb.images.class==ci);
        trainvalSids = unique(imdb.images.sid(inds));
        nValShapes = floor(opts.ratio(2)*numel(trainvalSids));
        trainvalSids = trainvalSids(randperm(numel(trainvalSids)));
        valSids = trainvalSids(1:nValShapes);
        inds = ismember(imdb.images.sid,valSids);
        imdb.images.set(inds) = 2;
        nTrainShapes = nTrainShapes - nValShapes;
    else
        nValShapes = 0;
    end
    
    % test
    if hasTest,
        files = dir(fullfile(testDir, ['*' opts.ext]));
        fileNames = {files.name};
        nTestImages = length(fileNames);
        imVids = cellfun(@(s) get_shape_vid(s), fileNames);
        [~,I] = sort(imVids);
        fileNames = fileNames(I); % order images wrt view id
        sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
            'UniformOutput', false);
        sNamesUniq = unique(sNames);
        nTestShapes = length(sNamesUniq);
        [~,imSids] = ismember(sNames,sNamesUniq);
        if isempty(imdb.images.sid), maxSid = 0;
        else maxSid = max(imdb.images.sid); end
        imdb.images.sid = [imdb.images.sid maxSid+imSids];
        imdb.images.set = [imdb.images.set 3*ones(1,nTestImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nTestImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},'test',s), ...
            fileNames, 'UniformOutput',false)];
    else
        nTestShapes = 0;
    end
    
    fprintf('\ttrain/val/test: %d/%d/%d (shapes)\n', ...
        nTrainShapes, nValShapes, nTestShapes);
end
end


function numShapes = render_meshes(imdb, opts)
for ci = 1:length(imdb.meta.classes),
    fprintf('  [%2d/%2d] %s ... ', ci, length(imdb.meta.classes), ...
        imdb.meta.classes{ci});
    trainDir = fullfile(imdb.imageDir,imdb.meta.classes{ci},'train');
    valDir = fullfile(imdb.imageDir,imdb.meta.classes{ci},'val');
    testDir = fullfile(imdb.imageDir,imdb.meta.classes{ci},'test');
    if exist(trainDir,'dir'), hasTrain = true; else hasTrain = false; end;
    if exist(valDir,'dir'), hasVal = true; else hasVal = false; end;
    if exist(testDir,'dir'), hasTest = true; else hasTest = false; end;
    
    mesh_filenames = {};
    if hasTrain,
        files = dir(fullfile(trainDir, ['*' opts.extmesh]));
        for fi=1:length(files)
            mesh_filenames{end+1} = fullfile(trainDir, files(fi).name);
        end
    end
    if hasVal,
        files = dir(fullfile(valDir, ['*' opts.extmesh]));
        for fi=1:length(files)
            mesh_filenames{end+1} = fullfile(valDir, files(fi).name);
        end        
    end
    if hasTest,
        files = dir(fullfile(testDir, ['*' opts.extmesh]));
        for fi=1:length(files)
            mesh_filenames{end+1} = fullfile(testDir, files(fi).name);
        end                
    end
    numShapes = length(mesh_filenames);
    fprintf('Found %d meshes.\n', numShapes);
    
    fig = figure('Visible','off');
    for fi=1:numShapes
        fprintf('Loading and rendering input shape %s...', mesh_filenames{fi});
        mesh = loadMesh( mesh_filenames{fi} );
        if isempty(mesh.F)
            error('Could not load mesh from file');
        else
            fprintf('Done.\n');
        end
        if opts.useUprightAssumption
            ims = render_views(mesh, 'figHandle', fig);
        else
            ims = render_views(mesh, 'use_dodecahedron_views', true, 'figHandle', fig);
        end
        
        for ij=1:length(ims)
            imwrite( ims{ij}, sprintf('%s_%03d%s', mesh_filenames{fi}(1:end-4), ij, opts.ext) );
        end
    end
    close(fig);

end
end


function shapename = get_shape_name(filename)
suffix_idx = strfind(filename,'_');
if isempty(suffix_idx),
    shapename = [];
else
    shapename = filename(1:suffix_idx(end)-1);
end
end


function vid = get_shape_vid(filename)
suffix_idx = strfind(filename,'_');
ext_idx = strfind(filename,'.');
if isempty(suffix_idx) || isempty(ext_idx),
    vid = [];
else
    vid = str2double(filename(suffix_idx(end)+1:ext_idx(end)-1));
end
end
