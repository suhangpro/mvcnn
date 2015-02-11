function [ imdb ] = get_imdb( datasetName )
%GET_IMDB Get imdb structure for the specified dataset
% datasetName 
%   should be name of a directory under '/data'

datasetDir = fullfile('data',datasetName);
datasetFnName = ['setup_' datasetName];
imdbPath = fullfile(datasetDir,'imdb.mat');

if ~exist(datasetDir,'dir') || ~exist([datasetFnName '.m'],'file'), 
    error('Unknown dataset: %s', datasetName);
end

if exist(imdbPath,'file'), 
    imdb = load(imdbPath);
else
    imdb = eval([datasetFnName '(''' datasetDir ''')']);
    save(imdbPath,'-struct','imdb');
end

end

