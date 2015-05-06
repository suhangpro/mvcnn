function imdb = setup_dataset(datasetDir, varargin)

opts.seed = 0 ;             % random seed generator
opts.ratio = [0.8 0.2];     % train:val ratio
opts.invert = false;        % if true, for each pixel: v --> 255-v
opts.ext = '.png';          % extension of target files
opts.useShape = true;       % if true, instances are grouped by shape id
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
imdb.meta.invert = opts.invert;
imdb.meta.sets = {'train', 'val', 'test'};
fprintf('%d classes found! \n', length(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.class   = [];
imdb.images.set     = [];
if opts.useShape, 
    imdb.images.sid = [];
end
fprintf('Scanning for images: \n');
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
        fileNames = fileNames(randperm(nTrainImages)); 
        if opts.useShape, 
            imVids = cellfun(@(s) get_shape_vid(s), fileNames);
            [~,I] = sort(imVids);
            fileNames = fileNames(I); % order images wrt view id
            sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
                'UniformOutput', false);
            sNamesUniq = unique(sNames);
            nTrainShapes = length(sNamesUniq);
            sNamesUniq = sNamesUniq(randperm(nTrainShapes));
            [~,imSids] = ismember(sNames,sNamesUniq);
            imdb.images.sid = [imdb.images.sid max(imdb.images.sid)+imSids];
        end
        imdb.images.set = [imdb.images.set ones(1,nTrainImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nTrainImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},'train',s), ...
            fileNames, 'UniformOutput',false)];
    else
        nTrainImages = 0;
        nTrainShapes = 0;
    end
    
    % val
    if hasVal,
        files = dir(fullfile(valDir, ['*' opts.ext]));
        fileNames = {files.name};
        nValImages = length(fileNames);
        fileNames = fileNames(randperm(nValImages));
        if opts.useShape,
            imVids = cellfun(@(s) get_shape_vid(s), fileNames);
            [~,I] = sort(imVids);
            fileNames = fileNames(I); % order images wrt view id
            sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
                'UniformOutput', false);
            sNamesUniq = unique(sNames);
            nValShapes = length(sNamesUniq);
            sNamesUniq = sNamesUniq(randperm(nValShapes));
            [~,imSids] = ismember(sNames,sNamesUniq);
            imdb.images.sid = [imdb.images.sid max(imdb.images.sid)+imSids];
        end
        imdb.images.set = [imdb.images.set 2*ones(1,nValImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nValImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},'val',s), ...
            fileNames, 'UniformOutput',false)];
    elseif hasTrain && opts.ratio(2)>0,
        if opts.useShape, 
            inds = (imdb.images.set==1 & imdb.images.class==ci);
            trainvalSids = unique(imdb.images.sid(inds));
            nValShapes = floor(opts.ratio(2)*numel(trainvalSids));
            valSids = trainvalSids(1:nValShapes);
            inds = ismember(imdb.images.sid,valSids);
            imdb.images.set(inds) = 2;
            nValImages = length(find(inds));
            nTrainShapes = nTrainShapes - nValShapes;
            nTrainImages = nTrainImages - nValImages;
        else
            idxs = find(imdb.images.set==1 & imdb.images.class==ci);
            nValImages = floor(opts.ratio(2)*length(idxs));
            imdb.images.set(idxs(1:nValImages)) = 2;
            nTrainImages = nTrainImages - nValImages;
        end
    else
        nValImages = 0;
        nValShapes = 0;
    end
    
    % test
    if hasTest, 
        files = dir(fullfile(testDir, ['*' opts.ext]));
        fileNames = {files.name}; 
        nTestImages = length(fileNames);
        fileNames = fileNames(randperm(nTestImages)); 
        if opts.useShape, 
            imVids = cellfun(@(s) get_shape_vid(s), fileNames);
            [~,I] = sort(imVids);
            fileNames = fileNames(I); % order images wrt view id
            sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
                'UniformOutput', false);
            sNamesUniq = unique(sNames);
            nTestShapes = length(sNamesUniq);
            sNamesUniq = sNamesUniq(randperm(nTestShapes));
            [~,imSids] = ismember(sNames,sNamesUniq);
            imdb.images.sid = [imdb.images.sid max(imdb.images.sid)+imSids];
        end
        imdb.images.set = [imdb.images.set 3*ones(1,nTestImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nTestImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},'test',s), ...
            fileNames, 'UniformOutput',false)];
    else
        nTestImages = 0;
        nTestShapes = 0;
    end
    
    if opts.useShape, 
        fprintf('\ttrain/val/test: %d/%d/%d (shapes)\n', ...
            nTrainShapes, nValShapes, nTestShapes);
    else
        fprintf('\ttrain/val/test: %d/%d/%d (images)\n', ...
            nTrainImages, nValImages, nTestImages);
    end
end
if opts.useShape, 
    [imdb.images.sid, I] = sort(imdb.images.sid);
    imdb.images.name = imdb.images.name(I);
    imdb.images.class = imdb.images.class(I);
    imdb.images.set = imdb.images.set(I);
end
imdb.images.id = 1:length(imdb.images.name);

function shapename = get_shape_name(filename)
suffix_idx = strfind(filename,'_');
if isempty(suffix_idx), 
    shapename = []; 
else
    shapename = filename(1:suffix_idx(end)-1);
end

function vid = get_shape_vid(filename)
suffix_idx = strfind(filename,'_');
ext_idx = strfind(filename,'.');
if isempty(suffix_idx) || isempty(ext_idx), 
    vid = []; 
else
    vid = str2double(filename(suffix_idx(end)+1:ext_idx(end)-1));
end
