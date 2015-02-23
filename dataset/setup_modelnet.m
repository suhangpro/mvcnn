function imdb = setup_modelnet(datasetDir, varargin)
% Set the random seed generator
opts.seed = 0 ;
opts.ratio = [0.8 0.2];
opts.suffix = 'toonedge';
opts.invert = true;
opts.nViews = 12;
opts = vl_argparse(opts, varargin);

rng(opts.seed) ;
opts.ratio = opts.ratio(1:2)/sum(opts.ratio(1:2));

imdb.imageDir = fullfile(datasetDir,'png');

% edge images need to be inverted (v --> 255-v)
imdb.meta.invert = opts.invert;

% images and class
imdb.meta.classes = {};
imdb.images.name = {};
imdb.images.class = [];

% get list of test files
fid = fopen(fullfile(datasetDir, 'modelnet-test.txt'),'r');
testListFull = textscan(fid, '%s','Delimiter','\n');
testListFull = cellfun(@(s) s(1:end-4), testListFull{1}, 'UniformOutput', false);
testListFull = reshape(testListFull,[1 length(testListFull)]);
fclose(fid);

% class list
files = dir(imdb.imageDir); 
for i=1:length(files), 
    if ~(strcmp(files(i).name,'.') || strcmp(files(i).name,'..')) ...
            && files(i).isdir
        imdb.meta.classes = [imdb.meta.classes files(i).name]; 
    end
end

% get list of trainval files
% split trainval randomly into train/val sets according to opts.ratio
trainList = {};
valList = {};
testList = {};
trainClass = [];
valClass = [];
testClass = [];
for c = 1:length(imdb.meta.classes), 
    className = imdb.meta.classes{c};
    fprintf('[%2d/%2d] %s ...', c,length(imdb.meta.classes),className);
    files = dir(fullfile(imdb.imageDir,className,'*.png'));
    shapes = unique(cellfun(@(str) get_shape_name(str,opts.suffix), ...
        {files.name}, 'UniformOutput', false));
    shapes = reshape(shapes, [1 length(shapes)]);
    
    trainval = setdiff(shapes,testListFull);        
    order = randperm(length(trainval));
    nTrain = ceil(length(order)*opts.ratio(1));
    train = trainval(order(1:nTrain));
    val = trainval(order(nTrain+1:end));
    test = intersect(testListFull,shapes);
    
    trainList = [trainList cellfun(@(s) fullfile(className,s), ...
        train, 'UniformOutput', false)]; %#ok<AGROW>
    trainClass = [trainClass c*ones(1,length(train))]; %#ok<AGROW>
    
    valList = [valList cellfun(@(s) fullfile(className,s), ...
        val, 'UniformOutput', false)]; %#ok<AGROW>
    valClass = [valClass c*ones(1,length(val))]; %#ok<AGROW>
    
    testList = [testList cellfun(@(s) fullfile(className,s), ...
        test, 'UniformOutput', false)]; %#ok<AGROW>
    testClass = [testClass c*ones(1,length(test))]; %#ok<AGROW>
    fprintf(' %5d shapes found ... done!\n', ... 
        length(train) + length(val) + length(test));
end

nShapes = length(trainList) + length(valList) + length(testList);
nImages = nShapes*opts.nViews;

% multiply for views
fprintf('Multipling for %d views ...', opts.nViews);
imdb.meta.sets = {'train', 'val', 'test'};
imdb.images.id = 1:nImages;
imdb.images.set = repmat([ones(1,length(trainList)), ...
    2*ones(1,length(valList)), 3*ones(1,length(testList))],[1 opts.nViews]);
imdb.images.class = repmat([trainClass valClass testClass],[1 opts.nViews]);
imdb.images.name = cell(1,nImages);
names = [trainList valList testList];
for v = 1:opts.nViews, 
    names_v = cellfun(@(s) sprintf('%s_%s_%d.png',s,opts.suffix,v), names, ...
        'UniformOutput', false);
    imdb.images.name((v-1)*nShapes+[1:nShapes]) = names_v; %#ok<NBRAK>
end
fprintf(' done!\n');

order = randperm(nImages);
imdb.images.name = imdb.images.name(order);
imdb.images.id = imdb.images.id(order);
imdb.images.class = imdb.images.class(order);
imdb.images.set = imdb.images.set(order);

function shapename = get_shape_name(filename,suffix)

suffix_idx = strfind(filename,['_' suffix]);
if isempty(suffix_idx), 
    shapename = []; 
else
    shapename = filename(1:suffix_idx-1);
end