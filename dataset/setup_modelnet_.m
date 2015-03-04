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
if exist(testDir, 'dir'),
    hasTest = true;
else
    hasTest = false;
end

% meta
folders = {};
fprintf('Scanning for classes ... ');
contents = dir(trainDir);
for i=1:numel(contents),
    if contents(i).isdir, folders = [folders contents(i).name]; end
end
if hasTest
    contents = dir(testDir);
    for i=1:numel(contents),
        if contents(i).isdir, folders = [folders contents(i).name]; end
    end
end
imdb.meta.classes = setdiff(unique(folders),{'.','..'});
imdb.meta.invert = opts.invert; % edges need to be inverted (v --> 255-v)
imdb.meta.sets = {'train', 'val', 'test'};
if ~hasTest, imdb.meta.sets{3} = []; end;
fprintf('%d classes found! \n', length(imdb.meta.classes));

% images
imdb.images.name = {};
imdb.images.class = [];
imdb.images.set = [];
fprintf('Scanning for images: \n');
for ci = 1:length(imdb.meta.classes),
    fprintf('  [%2d/%2d] %s ... ', ci, length(imdb.meta.classes), ...
        imdb.meta.classes{ci});
    % train (& val)
    files = dir(fullfile(trainDir, imdb.meta.classes{ci}, ['*' opts.ext]));
    nTrainval = length(files);
    fileNames = {files.name}; 
    shapeNames = cellfun(@(s) get_shape_name(s), fileNames, ...
        'UniformOutput', false); 
    shapeNamesUnique = unique(shapeNames); 
    [~,shapeIdxs] = ismember(shapeNames, shapeNamesUnique); 
    nShapes = length(shapeNamesUnique); 
    order = randperm(nShapes);
    nTrainShapes = ceil(nShapes*opts.ratio(1));
    val = order(nTrainShapes+1:end); 
    imagesSet = ones(1, nTrainval);
    imagesSet(ismember(shapeIdxs,val)) = 2;
    nTrain = sum(imagesSet==1); 
    nVal = nTrainval - nTrain; 
    imdb.images.set = [imdb.images.set imagesSet];
    imdb.images.name = [imdb.images.name cellfun(@(s) fullfile('training', ...
        imdb.meta.classes{ci}, s), fileNames, 'UniformOutput', false)];
    imdb.images.class = [imdb.images.class ci*ones(1,nTrainval)];
    % test
    nTest = 0;
    if hasTest,
        files = dir(fullfile(testDir, imdb.meta.classes{ci}, ['*' opts.ext]));
        nTest = length(files);
        imdb.images.name = [imdb.images.name cellfun(@(s) fullfile('testing', ...
            imdb.meta.classes{ci}, s), {files.name}, 'UniformOutput', false)];
        imdb.images.class = [imdb.images.class ci*ones(1,nTest)];
        imdb.images.set = [imdb.images.set 3*ones(1,nTest)];
    end
    fprintf('\ttrain/val/test: %d/%d/%d\n', nTrain, nVal, nTest);
end
imdb.images.id = 1:length(imdb.images.name);

% shuffle
fprintf('Shuffling ... ');
order = randperm(length(imdb.images.name));
imdb.images.name = imdb.images.name(order);
imdb.images.id = imdb.images.id(order);
imdb.images.class = imdb.images.class(order);
imdb.images.set = imdb.images.set(order);
fprintf('done! \n');

function shapename = get_shape_name(filename)

suffix_idx = strfind(filename,'_');
if isempty(suffix_idx), 
    shapename = []; 
else
    shapename = filename(1:suffix_idx(end)-1);
end
