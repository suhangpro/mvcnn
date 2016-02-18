function net = cnn_shape_init(classNames, varargin)
opts.base = 'imagenet-matconvnet-vgg-m'; 
opts.restart = false; 
opts.nViews = 12; 
opts.viewpoolPos = 'relu5'; 
opts.weightInitMethod = 'xavierimproved';
opts.scale = 1;
opts.networkType = 'simplenn'; % only simplenn is supported currently
opts = vl_argparse(opts, varargin); 

assert(strcmp(opts.networkType,'simplenn'), 'Only simplenn is supported currently'); 

init_bias = 0.1;
nClass = length(classNames);

% Load model, try to download it if not readily available
if ~ischar(opts.base), 
  net = opts.base; 
else
  netFilePath = fullfile('data','models', [opts.base '.mat']);
  if ~exist(netFilePath,'file'),
    fprintf('Downloading model (%s) ...', opts.base) ;
    vl_xmkdir(fullfile('data','models')) ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models/', ...
      [opts.base '.mat']), netFilePath) ;
    fprintf(' done!\n');
  end
  net = load(netFilePath);
end
assert(strcmp(net.layers{end}.type, 'softmax'), 'Wrong network format'); 
dataTyp = class(net.layers{end-1}.weights{1}); 

% Initiate the last but one layer w/ random weights
widthPrev = size(net.layers{end-1}.weights{1}, 3);
nClass0 = size(net.layers{end-1}.weights{1},4);
if nClass0 ~= nClass || opts.restart, 
  net.layers{end-1}.weights{1} = init_weight(opts, 1, 1, widthPrev, nClass, dataTyp);
  net.layers{end-1}.weights{2} = zeros(nClass, 1, dataTyp); 
end

% Initiate other layers w/ random weights if training from scratch is desired
if opts.restart, 
  w_layers = find(cellfun(@(c) isfield(c,'weights'),net.layers));
  for i=1:numel(w_layers)-1, 
    sz = size(net.layers{i}.weights{1}); 
    net.layers{i}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataTyp);
    net.layers{i}.weights{2} = zeros(sz(4), 1, dataTyp); 
  end	
end

% Swap softmax w/ softmaxloss
net.layers{end} = struct('type', 'softmaxloss', 'name', 'loss') ;
    
% Insert viewpooling
if opts.nViews>1, 
  viewpoolLayer = struct('name', 'viewpool', ...
    'type', 'custom', ...
    'vstride', opts.nViews, ...
    'method', 'max', ...
    'forward', @viewpool_fw, ...
    'backward', @viewpool_bw);
  net = modify_net(net, viewpoolLayer, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);
end

% update meta data
net.meta.classes.name = classNames;
net.meta.classes.description = classNames;
    
end


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end


% -------------------------------------------------------------------------
function res_ip1 = viewpool_fw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_i.x);
if mod(sz4,layer.vstride)~=0, 
    error('all shapes should have same number of views');
end
if strcmp(layer.method, 'avg'), 
    res_ip1.x = permute(...
        mean(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), 4), ...
        [1,2,3,5,4]);
elseif strcmp(layer.method, 'max'), 
    res_ip1.x = permute(...
        max(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), [], 4), ...
        [1,2,3,5,4]);
else
    error('Unknown viewpool method: %s', layer.method);
end

end


% -------------------------------------------------------------------------
function res_i = viewpool_bw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_ip1.dzdx);
if strcmp(layer.method, 'avg'), 
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx / layer.vstride, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]);
elseif strcmp(layer.method, 'max'), 
    [~,I] = max(reshape(permute(res_i.x,[4 1 2 3]), ...
                [layer.vstride, sz4*sz1*sz2*sz3]),[],1);
    Ind = zeros(layer.vstride,sz4*sz1*sz2*sz3, 'single');
    Ind(sub2ind(size(Ind),I,1:length(I))) = 1;
    Ind = permute(reshape(Ind,[layer.vstride*sz4,sz1,sz2,sz3]),[2 3 4 1]);
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]) .* Ind;
else
    error('Unknown viewpool method: %s', layer.method);
end

end
