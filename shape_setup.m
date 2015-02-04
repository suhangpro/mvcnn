function [opts, imdb] = shape_setup(varargin)
setup ;
opts.seed = 1 ;
opts.batchSize = 128 ;
opts.useGpu = true;
opts.printDatasetInfo = false ;
opts.dataset = 'sketch' ;
opts.sketchDir = 'data/sketch';
opts.clipartgpbDir = 'data/clipartgpb';
opts.model = 'imagenet-vgg-m.mat';
opts.prefix = 'v1' ;
opts.suffix = 'baseline' ; % NOT USED?
opts.regionBorder = 0.05 ; % NOT USED?
opts.excludeDifficult = true ; % NOT USED?
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
if ~exist(fullfile('data/models', opts.model))
  fprintf('downloading model %s\n', opts.model) ;
  vl_xmkdir('data/models') ;
  urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', opts.model),...
      fullfile('data/models', opts.model)) ;
end


% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------
vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;
imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
if exist(imdbPath, 'file'), 
    imdb = load(imdbPath) ;
else
    switch opts.dataset
        case 'sketch'
            imdb = sketch_get_database(opts.sketchDir, 'seed', opts.seed);
        case 'clipartgpb'
            imdb = clipart_get_database(opts.clipartgpbDir, 'seed', opts.seed);
        otherwise
            error('Unknown dataset %s', opts.dataset) ;
    end

    save(imdbPath, '-struct', 'imdb') ;
end

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end
