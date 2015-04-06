function imdb = setup_modelnet_(datasetDir, varargin)

% Set the random seed generator
opts.seed = 0 ;
opts.ratio = [0.8 0.2];
opts.invert = false;
opts.ext = '.png';
opts = vl_argparse(opts, varargin);

rng(opts.seed) ;
opts.ratio = opts.ratio(1:2)/sum(opts.ratio(1:2)); % train:val ratio

% imageDir
imdb.imageDir = datasetDir;
trainDir = fullfile(datasetDir, 'training');
testDir = fullfile(datasetDir, 'testing');

% meta
folders = {};
fprintf('Scanning for classes ... ');
contents = dir(trainDir);
for i=1:numel(contents),
    if contents(i).isdir, folders = [folders contents(i).name]; end
end
contents = dir(testDir);
for i=1:numel(contents),
    if contents(i).isdir, folders = [folders contents(i).name]; end
end
imdb.meta.classes = setdiff(unique(folders),{'.','..'});
imdb.meta.invert = opts.invert; % edges need to be inverted (v --> 255-v)
imdb.meta.sets = {'train', 'val', 'test'};
fprintf('%d classes found! \n', length(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.class   = [];
imdb.images.set     = [];
imdb.images.sid     = [];
cntShapes = 0;
fprintf('Scanning for images: \n');
for ci = 1:length(imdb.meta.classes),
    fprintf('  [%2d/%2d] %s ... ', ci, length(imdb.meta.classes), ...
        imdb.meta.classes{ci});
    % train (& val)
    files = dir(fullfile(trainDir, imdb.meta.classes{ci}, ['*' opts.ext]));
    nTrainval = length(files);
    fileNames = {files.name}; 
    fileNames = fileNames(randperm(nTrainval)); % shuffle images 
    imVids = cellfun(@(s) get_shape_vid(s), fileNames);
    [~, I] = sort(imVids);
    fileNames = fileNames(I); % order images wrt view id
    imSnames = cellfun(@(s) get_shape_name(s), fileNames, ...
        'UniformOutput', false); 
    snamesUnique = unique(imSnames);  
    nShapes = length(snamesUnique); 
    snamesUnique = snamesUnique(randperm(nShapes)); % shuffle shape id
    [~,imSids] = ismember(imSnames, snamesUnique);
    nValShapes = ceil(nShapes*opts.ratio(2));
    imSet = ones(1, nTrainval);
    imSet(ismember(imSids,1:nValShapes)) = 2;
    nTrain = sum(imSet==1); 
    nVal = nTrainval - nTrain; 
    imdb.images.set = [imdb.images.set imSet];
    imdb.images.name = [imdb.images.name cellfun(@(s) fullfile('training', ...
        imdb.meta.classes{ci}, s), fileNames, 'UniformOutput', false)];
    imdb.images.class = [imdb.images.class ci*ones(1,nTrainval)];
    imdb.images.sid = [imdb.images.sid imSids+cntShapes];
    cntShapes = cntShapes + nShapes; 
    % test
    files = dir(fullfile(testDir, imdb.meta.classes{ci}, ['*' opts.ext]));
    nTest = length(files);
    fileNames = {files.name};
    fileNames = fileNames(randperm(nTest)); % shuffle images 
    imVids = cellfun(@(s) get_shape_vid(s), fileNames);
    [~, I] = sort(imVids);
    fileNames = fileNames(I); % order images wrt view id
    imSnames = cellfun(@(s) get_shape_name(s), fileNames, ...
        'UniformOutput', false);
    snamesUnique = unique(imSnames);
    nShapes = length(snamesUnique); 
    snamesUnique = snamesUnique(randperm(nShapes)); % shuffle shape id
    [~,imSids] = ismember(imSnames, snamesUnique);
    imdb.images.set = [imdb.images.set 3*ones(1,nTest)];
    imdb.images.name = [imdb.images.name cellfun(@(s) fullfile('testing', ...
        imdb.meta.classes{ci}, s), fileNames, 'UniformOutput', false)];
    imdb.images.class = [imdb.images.class ci*ones(1,nTest)];
    imdb.images.sid = [imdb.images.sid imSids+cntShapes];
    cntShapes = cntShapes + nShapes;
    
    fprintf('\ttrain/val/test: %d/%d/%d\n', nTrain, nVal, nTest);
end
[imdb.images.sid, I] = sort(imdb.images.sid);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);
imdb.images.id = 1:length(imdb.images.name);
imdb.meta.nShapes = cntShapes;

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
