function [opts, imdb] = shape_setup(varargin)
setup ;
opts.seed = 1 ;
opts.batchSize = 128 ;
opts.useGpu = true ;
opts.regionBorder = 0.05 ;
opts.printDatasetInfo = false ;
opts.excludeDifficult = true ;
opts.dataset = 'os' ;
opts.vocDir = 'data/VOC2007' ;
opts.sketchDir = 'data/sketch';
opts.suffix = 'baseline' ;
opts.prefix = 'v1' ;
opts.model = 'imagenet-vgg-m.mat';
[opts, varargin] = vl_argparse(opts,varargin) ;

opts.expDir = sprintf('data/%s/%s-seed-%02d', opts.prefix, opts.dataset, opts.seed) ;
opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
opts = vl_argparse(opts,varargin) ;

if nargout <= 1, return ; end

% Setup GPU if needed
if opts.useGpu
  gpuDevice(1) ;
end

% -------------------------------------------------------------------------
%                                                       Download CNN models
% -------------------------------------------------------------------------
for i = 1:numel(models)
  if ~exist(fullfile('data/models', models{i}))
    fprintf('downloading model %s\n', models{i}) ;
    vl_xmkdir('data/models') ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', models{i}),...
      fullfile('data/models', models{i})) ;
  end
end

% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------
vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;
imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
if exist(imdbPath, 'file')
  imdb = load(imdbPath) ;
  return ;
end

switch opts.dataset
  case 'sketch'
    imdb = sketch_get_database(opts.sketchDir, 'seed', opts.seed);
  otherwise
    error('Unknown dataset %s', opts.dataset) ;
end

save(imdbPath, '-struct', 'imdb') ;

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end