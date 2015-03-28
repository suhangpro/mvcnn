function imdb = setup_clipart_(clipartDir, varargin)
% Set the random seed generator
opts.seed = 0 ;
opts.ratio = [0.5 0.2 0.3];
opts.invert = false;
opts.limitPerClass = 100 ;
opts = vl_argparse(opts, varargin) ;
rng(opts.seed) ;

imdb.imageDir = fullfile(clipartDir);
fid = fopen(fullfile(clipartDir, 'classlist.txt'));
classlist = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

imdb.meta.invert = opts.invert;

% Images and class
imdb.meta.classes = classlist{1}';
imdb.images.name = {};
for c = imdb.meta.classes, 
    c = c{:};
    filelist = dir(fullfile(clipartDir,c,'*.png'));
    imagePaths = cellfun(@(x) fullfile(c,x),{filelist.name},'UniformOutput',false);
    slctIdx = randperm(length(imagePaths));
    imagePaths = imagePaths(slctIdx(1:min(opts.limitPerClass,length(slctIdx))));
    imdb.images.name = [imdb.images.name,imagePaths];
end
imdb.images.id = 1:length(imdb.images.name);
class = cellfun(@(x) fileparts(x), imdb.images.name, 'UniformOutput', false);

% Class names
[~, imdb.images.class] = ismember(class, imdb.meta.classes);

% No standard image splits are provided for this dataset, so split them
% randomly into train/val/test sets according to opts.ratio
imdb.meta.sets = {'train', 'val', 'test'};
imdb.images.set = zeros(1,length(imdb.images.id));
for c = 1:length(imdb.meta.classes), 
    isclass = find(imdb.images.class == c);
    
    % split equally into train, val, test
    order = randperm(length(isclass));
    subsetSizeTrain = ceil(length(order)*opts.ratio(1));
    subsetSizeVal = ceil(length(order)*opts.ratio(2));
    train = isclass(order(1:subsetSizeTrain));
    val = isclass(order(subsetSizeTrain+1:subsetSizeTrain+subsetSizeVal));
    test  = isclass(order(subsetSizeTrain+subsetSizeVal+1:end));
    
    imdb.images.set(train) = 1;
    imdb.images.set(val) = 2;
    imdb.images.set(test) = 3;
end

% shuffle
order = randperm(length(imdb.images.name));
imdb.images.name = imdb.images.name(order);
% imdb.images.id = imdb.images.id(order);
imdb.images.class = imdb.images.class(order);
imdb.images.set = imdb.images.set(order);
