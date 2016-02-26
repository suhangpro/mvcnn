function [ imdb ] = get_imdb( datasetName, varargin )
%GET_IMDB Get imdb structure for the specified dataset
% datasetName 
%   should be name of a directory under '/data'
% 'func'
%   the function that actually builds the imdb 
%   default: @setup_imdb_generic
% 'rebuild'
%   whether to rebuild imdb if one exists already
%   default: false


args.func = @setup_imdb_generic;
args.rebuild = false;
args = vl_argparse(args,varargin);

datasetDir = fullfile('data',datasetName);
imdbPath = fullfile(datasetDir,'imdb.mat');

if ~exist(datasetDir,'dir'), 
    error('Unknown dataset: %s', datasetName);
end

if exist(imdbPath,'file') && ~args.rebuild, 
    fprintf('Loading imdb from %s ...', imdbPath);
    imdb = load(imdbPath);
    fprintf(' done!\n');
else
    imdb = args.func(datasetDir);
    save(imdbPath,'-struct','imdb');
end

end

