setup;

if ~exist('feat','var'), 
  feat = load('data/features/modelnet40phong-imagenet-vgg-m-finetuned-modelnet40phong-BS60_AUGnone-finetuned-modelnet40phong-BS60_AUGnone_MVfc7-none/NORM0/relu7.mat');
  % feat = load('data/features/modelnet40phong-imagenet-vgg-m-finetuned-modelnet40phong-BS60_AUGnone-none/NORM0/relu7.mat');
end

imdb = feat.imdb;
if isfield(feat,'sid'), 
  [imdb.images.sid,I] = sort(imdb.images.sid);
  imdb.images.id = imdb.images.id(I);
else
  [imdb.images.id,I] = sort(imdb.images.id);
end
imdb.images.set = imdb.images.set(I);
imdb.images.class = imdb.images.class(I);
imdb.images.name = imdb.images.name(I);

if isfield(feat,'sid'), 
  shapeStartIdx = find(imdb.images.sid-[Inf imdb.images.sid(1:end-1)]);
  shapeSid = imdb.images.sid(shapeStartIdx);
  [feat.sid, I] = sort(feat.sid);
  feat.x = feat.x(I,:);
  assert(isequal(feat.sid,shapeSid'));
else
  shapeStartIdx = 1:numel(imdb.images.id);
  [feat.id, I] = sort(feat.id);
  feat.x = feat.x(I,:);
  assert(isequal(feat.id,imdb.images.id'));
end
shapeSet = imdb.images.set(shapeStartIdx);
shapeClass = imdb.images.class(shapeStartIdx);
trainInd = shapeSet==1 | shapeSet==2;
trainClass = shapeClass(trainInd);

trainX = feat.x(trainInd,:);
classIds = unique(trainClass);

params.targetDim = 128;
if isfield(feat,'id'), params.numPairs = 1e7; 
else params.numPairs = 1e6; end
params.W = [];
params.b = [];


% TODO - hyperparameter grid-search by cross-validation
params.gammaBias = 0.001; 
params.gamma = 0.0001;
params.lambda = 0;

% The projection matrix is modelDimRedFV.W
tic
modelDimRedFV = trainProj( trainX', trainClass, classIds, params );
toc



