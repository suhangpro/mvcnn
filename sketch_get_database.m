function imdb = sketch_get_database(sketchDir, varargin)
% Set the random seed generator
opts.seed = 0 ;
opts = vl_argparse(opts, varargin) ;
rng(opts.seed) ;

imdb.imageDir = fullfile(sketchDir);
fid = fopen(fullfile(sketchDir, 'filelist.txt'));
filelist = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

% Images and class
imdb.images.name = filelist{1}';
imdb.images.id = 1:length(imdb.images.name);
class = cellfun(@(x) fileParts(x), imdb.images.name, 'UniformOutput', false);

% Class names
classNames = unique(class);
imdb.classes.name = classNames;
[~, imdb.images.label] = ismember(class, classNames);

% No standard image splits are provided for this dataset, so split them
% randomly into equal sized train/val/test sets
imdb.sets = {'train', 'val', 'test'};
imdb.images.set = zeros(1,length(imdb.images.id));
for c = 1:length(imdb.classes.name), 
    isclass = find(imdb.images.label == c);
    
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