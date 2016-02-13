function imdb = setup_imdb_generic(datasetDir, varargin)

opts.seed = 0 ;             % random seed generator
opts.ratio = [0.7 0.3];     % train:val ratio
opts.ext = '.jpg';          % extension of target files
opts.per_class_limit = inf; % inf indicates no limit
opts = vl_argparse(opts, varargin);

opts.ratio = opts.ratio(1:2)/sum(opts.ratio(1:2));
rng(opts.seed);
imdb.imageDir = datasetDir;

% meta
fprintf('Scanning for classes ... ');
contents = dir(imdb.imageDir);
contents_name = {contents.name};
imdb.meta.classes = setdiff(contents_name(cell2mat({contents.isdir})),{'.','..'});
imdb.meta.sets = {'train', 'val', 'test'};
fprintf('%d classes found! \n', numel(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.class   = [];
imdb.images.set     = []; 
fprintf('Scanning for images: \n');
for c = imdb.meta.classes, 
  c = c{1};
  fprintf('\t%s ...', c);
  contents = dir(fullfile(imdb.imageDir,c,['*' opts.ext]));
  curr_name = cellfun(@(s) fullfile(c,s),{contents.name},'UniformOutput',false);
  curr_name = curr_name(1:min(numel(curr_name),opts.per_class_limit));
  imdb.images.name = [imdb.images.name curr_name]; 
  imdb.images.class = [imdb.images.class ones(1,numel(curr_name))*find(strcmp(imdb.meta.classes,c))];
  curr_set = ones(1,floor(opts.ratio(1)*numel(curr_name)));
  curr_set = [curr_set 2*ones(1,numel(curr_name)-numel(curr_set))];
  imdb.images.set = [imdb.images.set curr_set(randperm(numel(curr_set)))];
  fprintf(' done!\n');
end

% id
imdb.images.id = 1:length(imdb.images.name);

