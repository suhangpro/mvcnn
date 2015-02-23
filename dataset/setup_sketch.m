function imdb = setup_sketch(sketchDir, varargin)
% Set the random seed generator
opts.seed = 0 ;
opts = vl_argparse(opts, varargin) ;
rng(opts.seed) ;

imdb.imageDir = fullfile(sketchDir,'png');
fid = fopen(fullfile(sketchDir, 'png', 'filelist.txt'));
filelist = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

% sketch images need to be inverted (v --> 255-v)
imdb.meta.invert = true;

% Images and class
imdb.images.name = filelist{1}';
imdb.images.id = 1:length(imdb.images.name);
class = cellfun(@(x) fileparts(x), imdb.images.name, 'UniformOutput', false);

% Class names
classNames = unique(class);
imdb.meta.classes = classNames;
[~, imdb.images.class] = ismember(class, classNames);

% No standard image splits are provided for this dataset, so split them
% randomly into equal sized train/val/test sets
imdb.meta.sets = {'train', 'val', 'test'};
imdb.images.set = zeros(1,length(imdb.images.id));
for c = 1:length(imdb.meta.classes), 
    isclass = find(imdb.images.class == c);
    
    % split equally into train, val, test
    order = randperm(length(isclass));
    subsetSize = ceil(length(order)/3);
    train = isclass(order(1:subsetSize));
    val = isclass(order(subsetSize+1:2*subsetSize));
    test  = isclass(order(2*subsetSize+1:end));
    
    imdb.images.set(train) = 1;
    imdb.images.set(val) = 2;
    imdb.images.set(test) = 3;
end

% shuffle
order = randperm(length(imdb.images.name));
imdb.images.name = imdb.images.name(order);
imdb.images.id = imdb.images.id(order);
imdb.images.class = imdb.images.class(order);
imdb.images.set = imdb.images.set(order);
