function imdb = setup_3ddrawing(datasetDir, varargin)
% Set the random seed generator
opts.seed = 0 ;
opts.ratio = [0.6 0.2 0.2];
opts = vl_argparse(opts, varargin) ;
rng(opts.seed) ;

nViews = 12;

imdb.imageDir = fullfile(datasetDir,'pngs');

% edge images need to be inverted (v --> 255-v)
imdb.meta.invert = true;

% images and class
imdb.meta.classes = {};
imdb.images.name = {};
imdb.images.class = [];
metaFiles = dir(fullfile(imdb.imageDir,'*_metadata.txt'));

for m = 1:numel(metaFiles), 
    imageNameMajor = metaFiles(m).name(1:end-13);
    fid = fopen(fullfile(imdb.imageDir,metaFiles(m).name)); 
    classAssignMat = textscan(fid,'%d %s','Delimiter','\n');
    fclose(fid);
    idx = find(classAssignMat{1},1);
    if isempty(idx), continue; end;
    % imageFiles = dir(fullfile(imdb.imageDir,[imageNameMajor '_toonedge_*.png']));
    % imdb.images.name = [imdb.images.name {imageFiles.name}];
    for j = 1:nViews, 
        imdb.images.name = [imdb.images.name [imageNameMajor '_toonedge_' int2str(j) '.png']]; 
    end
    className = classAssignMat{2}{idx};
    [~,classIdx] = ismember(className,imdb.meta.classes);
    if classIdx==0, % first see
        imdb.meta.classes = [imdb.meta.classes className];
        classIdx = length(imdb.meta.classes);
    end
    % imdb.images.class(end+1:end+length(imageFiles)) = classIdx;
    imdb.images.class(end+1:end+nViews) = classIdx;
    if mod(m,20)==0, fprintf('.'); end
    if mod(m,1000)==0, fprintf('\t %d/%d\n', m, numel(metaFiles)); end
end
fprintf(' done!\n');
imdb.images.id = 1:length(imdb.images.name);

% No standard image splits are provided for this dataset, so split them
% randomly into train/val/test sets according to opts.ratio
imdb.meta.sets = {'train', 'val', 'test'};
imdb.images.set = zeros(1,length(imdb.images.id));
for c = 1:length(imdb.meta.classes), 
    isclass = find(imdb.images.class == c);
    
    % split equally into train, val, test
    order = randperm(length(isclass));
    nTrain = ceil(length(order)*opts.ratio(1));
    nVal = ceil(length(order)*opts.ratio(2));
    train = isclass(order(1:nTrain));
    val = isclass(order(nTrain+1:nTrain+nVal));
    test  = isclass(order(nTrain+nVal+1:end));
    
    imdb.images.set(train) = 1;
    imdb.images.set(val) = 2;
    imdb.images.set(test) = 3;
end
