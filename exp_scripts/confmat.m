addpath(genpath('/scratch1/Hang/Dropbox/tools/piotr_toolbox/toolbox'));
load('~/Desktop/pred.mat');
imdb = get_imdb('modelnet40toon');
inds = find(imdb.images.set==3);inds = inds(1:12:end);gt=imdb.images.class(inds)';
CM = confMatrix(gt,predTest,40);
confMatrixShow(CM,strrep(imdb.meta.classes,'_',' '),{'FontSize',10});
