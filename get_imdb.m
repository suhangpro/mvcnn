function [ imdb ] = get_imdb( datasetName, varargin )
%GET_IMDB Get imdb structure for the specified dataset
% datasetName 
%   should be name of a directory under '/data'

datasetDir = fullfile('data',datasetName);
datasetFnName = ['setup_' datasetName];
imdbPath = fullfile(datasetDir,'imdb.mat');

if ~exist(datasetDir,'dir'), 
    error('Unknown dataset: %s', datasetName);
end

if exist(imdbPath,'file'), 
    imdb = load(imdbPath);
else
    if exist([datasetFnName '.m'],'file'),
        imdb = eval([datasetFnName '(''' datasetDir ''')']);
    else
        imdb = setup_dataset(datasetDir, varargin{:});
    end
    save(imdbPath,'-struct','imdb');
end

end

