function imo = cnn_shape_get_batch(images, varargin)
% Modified from CNN_IMAGENET_GET_BATCH
% 
% - added `pad` option
% - deals with images of types other than jpeg
% - augmentation made consistent across views 

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.pad = 0;  % [TOP BOTTOM LEFT RIGHT]
opts.nViews = 1; 
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

% if only one value is given, apply the same amount of padding to all borders
if numel(opts.pad)==1, opts.pad = repmat(opts.pad,[1 4]); end
if numel(opts.border)==1, opts.border = repmat(opts.border,[1 2]); end

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(images) >= 1 && ischar(images{1}) ;

% isjpg is true if all images to fetch are of jpeg format
isjpg = fetch && strcmpi(images{1}(end-3:end),'.jpg'); 

assert(mod(numel(images),opts.nViews)==0, '''nViews'' is incompatible with input');
nViews = opts.nViews;
nShapes = numel(images)/nViews; 

if opts.prefetch
  if isjpg, vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch'); end
  imo = [] ;
  return ;
end

if fetch
  if isjpg, 
    im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
  else
    im = cell(size(images)); 
  end
else
  im = images ;
end

tfs = [] ;
switch opts.transformation
  case 'none'
    tfs = [
      .5 ;
      .5 ;
       0 ] ;
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case 'stretch'
  otherwise
    error('Uknown transformations %s', opts.transformation) ;
end
[~,transformations] = sort(rand(size(tfs,2), numel(images)), 1) ;

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
            numel(images)*opts.numAugments, 'single') ;

for i=1:nShapes, 
  for j=1:nViews, 
    % acquire image
    idx = (i-1)*nViews + j;  
    if isempty(im{idx})
      imt = imread(images{idx}) ;
      imt = single(imt) ; % faster than im2single (and multiplies by 255)
    else
      imt = im{idx} ;
    end
    if size(imt,3) == 1
      imt = cat(3, imt, imt, imt) ;
    end
    if j==1, 
      imArr = zeros(size(imt,1),size(imt,2),3,nViews,'single');
    end
    imArr(:,:,:,j) = imt; 
  end

  % pad
  if ~isempty(opts.pad) && any(opts.pad>0), 
    w = size(imArr,2);
    h = size(imArr,1); 
    imArrTmp = imArr; 
    imArr = 255*ones(h+sum(opts.pad(1:2)), w+sum(opts.pad(3:4)), 3, nViews, 'single'); 
    imArr(opts.pad(1)+(1:h), opts.pad(3)+(1:w),:,:) = imArrTmp; 
  end

  % resize
  w = size(imArr,2) ;
  h = size(imArr,1) ;
  factor = [(opts.imageSize(1)+opts.border(1))/h ...
            (opts.imageSize(2)+opts.border(2))/w];

  if opts.keepAspect
    factor = max(factor) ;
  end
  if any(abs(factor - 1) > 0.0001)
    imArr = imresize(imArr, ...
                     'scale', factor, ...
                     'method', opts.interpolation) ;
  end

  % crop & flip
  w = size(imArr,2) ;
  h = size(imArr,1) ;
  for ai = 1:opts.numAugments
    switch opts.transformation
      case 'stretch'
        sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
        dx = randi(w - sz(2) + 1, 1) ;
        dy = randi(h - sz(1) + 1, 1) ;
        flip = rand > 0.5 ;
      otherwise
        tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
        sz = opts.imageSize(1:2) ;
        dx = floor((w - sz(2)) * tf(2)) + 1 ;
        dy = floor((h - sz(1)) * tf(1)) + 1 ;
        flip = tf(3) ;
    end
    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    if flip, sx = fliplr(sx) ; end

    if ~isempty(opts.averageImage)
      offset = opts.averageImage ;
      if ~isempty(opts.rgbVariance)
        offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
      end
      imo(:,:,:,(ai-1)*numel(images)+(i-1)*nViews+(1:nViews)) = bsxfun(@minus, imArr(sy,sx,:,:), offset) ;
    else
      imo(:,:,:,(ai-1)*numel(images)+(i-1)*nViews+(1:nViews)) = imArr(sy,sx,:,:) ;
    end
  end
end
