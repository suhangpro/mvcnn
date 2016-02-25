function imdb = setup_imdb_shapenet(datasetDir, varargin)

opts.ext = '.jpg';          % extension of target files
opts.nViews = 80;
opts.useSubclass = false; 
opts.trainFile = fullfile(datasetDir,'train.csv');
opts.valFile = fullfile(datasetDir,'val.csv');
opts = vl_argparse(opts, varargin);

imdb.imageDir = datasetDir;

trainAnno = csvread(opts.trainFile, 1); 
valAnno = csvread(opts.valFile, 1); 
assert(numel(unique([trainAnno(:,1);valAnno(:,1)])) ...
  ==size(trainAnno,1)+size(valAnno,1)); 

if opts.useSubclass, 
  labelCol = 3; 
else
  labelCol = 2;
end

classIds = sort(unique([trainAnno(:,labelCol); valAnno(:,labelCol)])'); 
[~,trainAnno(:,labelCol)] = ismember(trainAnno(:,labelCol),classIds); 
[~,valAnno(:,labelCol)] = ismember(valAnno(:,labelCol),classIds); 

% meta
imdb.meta.classes = arrayfun(@(i) sprintf('%08d',i),classIds,'UniformOutput',false); 
imdb.meta.sets = {'train', 'val', 'test'};

% images
imdb.images.name    = {};
imdb.images.class   = [];
imdb.images.set     = [];
imdb.images.sid     = [];

% train
fprintf('Scanning for training images ... ');
[imdb, nTrainShapes] = add_to_imdb(imdb, 'train', 1, trainAnno, opts); 
fprintf('done! %d shapes found.\n', nTrainShapes); 
% val
fprintf('Scanning for validation images ... ');
[imdb, nValShapes] = add_to_imdb(imdb, 'val', 2, valAnno, opts); 
fprintf('done! %d shapes found.\n', nValShapes); 
% test
fprintf('Scanning for testing images ... ');
[imdb, nTestShapes] = add_to_imdb(imdb, 'test', 3, [], opts); 
fprintf('done! %d shapes found.\n', nTestShapes); 

imdb.images.id = 1:numel(imdb.images.name); 

function [imdb, nShapesAdded] = add_to_imdb(imdb, subDir, setId, anno, opts); 
if opts.useSubclass, 
  labelCol = 3; 
else
  labelCol = 2;
end
files = dir(fullfile(imdb.imageDir,subDir,['*' opts.ext]));
files = {files.name};
sids = cellfun(@get_shape_id, files); 
vids = cellfun(@get_shape_vid, files); 
[~, I] = sort(vids); 
files = files(I); 
[sids, I] = sort(sids(I)); 
files = files(I);
sids0 = sids(1:opts.nViews:end); 
assert(numel(sids0)==numel(unique(sids))); 
if isempty(anno), % test
  nShapesAdded = numel(sids0);
  label = -1*ones(1,opts.nViews*nShapesAdded);
else
  [I I2] = ismember(sids0,anno(:,1)');
  nShapesAdded = sum(I); 
  label = anno(I2(I),labelCol); 
  label = repmat(label', [opts.nViews 1]);
  label = label(:)'; 
  I = repmat(I, [opts.nViews 1]); 
  I = I(:)'; 
  files = files(I);
  sids = sids(I); 
end
imdb.images.name = [imdb.images.name cellfun(@(s) fullfile(subDir,s), files, 'UniformOutput', false)];
imdb.images.sid = [imdb.images.sid sids]; 
imdb.images.class = [imdb.images.class label]; 
imdb.images.set = [imdb.images.set setId*ones(1,opts.nViews*nShapesAdded)]; 
if ~isempty(anno) && size(anno,1)~=nShapesAdded, 
  warning('%d %s shapes not found', size(anno,1)-nShapesAdded, subDir); 
end

function sid = get_shape_id(filename)
suffix_idx = strfind(filename,'_');
if numel(suffix_idx)~=2,
    sid = [];
else
    sid = str2double(filename(suffix_idx(1)+1:suffix_idx(2)-1));
end

function vid = get_shape_vid(filename)
suffix_idx = strfind(filename,'_');
ext_idx = strfind(filename,'.');
if isempty(suffix_idx) || isempty(ext_idx),
    vid = [];
else
    vid = str2double(filename(suffix_idx(end)+1:ext_idx(end)-1));
end
