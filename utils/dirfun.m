function dirfun( dir_path, processFn, save_path, imreadFn, file_pattern, save_pattern, cnt_limit )
% dirfun Apply a function to each file in the directory
% 
% dir_path:     directory containing images, will be searched recursively 
% processFn:    function that will be applied to each image found
% save_path:    (default:: dir_path) path to save resized images
% imreadFn:     (default:: @imread_safe) function used to load images
% file_pattern: (default:: '*') images that will be processed
% save_pattern: (default:: '') e.g. '%02d.png' will save images as 01.png, 02.png, ...
% cnt_limit:    (default:: inf) the maximum number of images used in each folder itself
% 
% Hang Su

% if save_path is not specified, overwrite original image 
if ~exist('save_path','var') || isempty(save_path), 
  save_path = dir_path;
end

% default imreadFn
if ~exist('imreadFn','var') || isempty(imreadFn), 
  imreadFn = @imread_safe;
end

% default file pattern
if ~exist('file_pattern','var') || isempty(file_pattern), 
  file_pattern = '*';
end

% default save pattern
if ~exist('save_pattern','var') || isempty(save_pattern), 
  save_pattern = '';
end

% default cnt_limit
if ~exist('cnt_limit','var') || isempty(cnt_limit), 
  cnt_limit = inf;
end

if ischar(file_pattern), 
  file_pattern = {file_pattern};
end

% run recursively
update_dir(dir_path, save_path, processFn, imreadFn, file_pattern, save_pattern, cnt_limit); 

end

function update_dir(cur_dir, cur_save_dir, processFn, imreadFn, file_pattern, save_pattern, cnt_limit)

file_names = {};
for i = 1:numel(file_pattern), 
  files = dir(fullfile(cur_dir, file_pattern{i}));
  file_names_cur = {files.name}; 
  file_names_cur = file_names_cur(~cell2mat({files.isdir}));
  file_names = [file_names file_names_cur];
end
dirs = dir(cur_dir);
dir_names = {dirs.name};
dir_names = setdiff(dir_names(cell2mat({dirs.isdir})),{'.','..'});

if~exist(cur_save_dir,'dir'), mkdir(cur_save_dir); end;

% do work
im_cnt = 0; 
for i=1:numel(file_names), 
  im = imreadFn(fullfile(cur_dir, file_names{i})); 
  if isempty(im),
    if strcmp(cur_dir, cur_save_dir), 
      delete(fullfile(cur_dir, file_names{i}));
    end
    continue; 
  end;
  im = processFn(im); 
  im_cnt = im_cnt + 1;
  
  % save (possibly overwriting original image) 
  if isempty(save_pattern), 
    imwrite(im,fullfile(cur_save_dir, file_names{i})); 
  else
    if ~isempty(strfind(save_pattern,'%s')), 
      [~,cur_name] = fileparts(file_names{i}); 
      cur_name = strrep(save_pattern,'%s',cur_name);
    else
      cur_name = save_pattern;
    end
    if ~isempty(strfind(cur_name,'%d')), 
      cur_name = sprintf(cur_name,im_cnt);
    end
    imwrite(im,fullfile(cur_save_dir, cur_name)); 
  end
  
  if im_cnt >= cnt_limit, break; end
  
end

for d = 1:numel(dir_names), 
  update_dir(fullfile(cur_dir,dir_names{d}), ...
    fullfile(cur_save_dir,dir_names{d}), processFn, imreadFn, file_pattern, save_pattern, cnt_limit);
end

end

function im = imread_safe(path)
  try 
    im = imread(path);
  catch
    warning('Unable to load image: %s', path);
    im = [];
  end
end
