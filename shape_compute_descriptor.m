function descr = shape_compute_descriptor( path_to_shape, varargin )
% SHAPE_COMPUTE_DESCRIPTOR Compute and save the CNN descriptor for a single
% input obj/off shape, or for each shape in an input folder
% Quick examples of use:
% shape_compute_descriptor('bunny.off');
% shape_compute_descriptor('my_bunnies_folder/');
%
%   path_to_shape:: (default) 'data/'
%        can be either a filename for a mesh in OBJ/OFF format
%        or a name of folder containing multiple OBJ/OFF meshes
%   `cnnModel`:: (default) ''
%       this is a matlab file with the saved CNN parameters
%       if the default file is not found, it will be downloaded from our
%       server.
%       Note: The default mat file assumes that shapes that are
%       upright oriented according to +Z axis!
%       if you want to use the CNN trained *without upright assumption*, use
%       'cnn-vggm-relu7-v2'
%   `applyMetric`:: (default) false
%       set to true to disable transforming descriptor based on specified
%       distance metric
%   `metricModel`:: (default:) ''
%       this is a matlab file with the saved metric parameters
%       if the default file is not found, it will attempt to download from our
%       server
%   `gpus`:: (default) []
%       set to use GPU

setup;

if nargin<1 || isempty(path_to_shape),
    path_to_shape = 'data/';
end

% default options
opts.feature = 'viewpool'; 
opts.useUprightAssumption = true; 
opts.applyMetric = false;
opts.gpus = [];
[opts, varargin] = vl_argparse(opts,varargin);

opts.metricModel = '';
opts.cnnModel = '';
opts = vl_argparse(opts,varargin);

% locate network file
default_viewpool_loc = 'relu7'; 
vl_xmkdir(fullfile('data','models')) ;
if isempty(opts.cnnModel), 
    if opts.useUprightAssumption, 
        opts.cnnModel = sprintf('cnn-vggm-%s-v1',default_viewpool_loc);
        nViews = 12;
    else
        opts.cnnModel = sprintf('cnn-vggm-%s-v2',default_viewpool_loc);
        nViews = 80;
    end
    baseModel = 'imagenet-matconvnet-vgg-m';
    cnn = cnn_shape_init({},'base',baseModel,'viewpoolPos',default_viewpool_loc,'nViews',nViews);
    netFilePath = fullfile('data','models',[opts.cnnModel '.mat']);
    save(netFilePath,'-struct','cnn');
end

if ~ischar(opts.cnnModel), 
    cnn = opts.cnnModel;
else
    netFilePath = fullfile('data','models',[opts.cnnModel '.mat']);
    if ~exist(netFilePath,'file'),
        fprintf('Downloading model (%s) ...', opts.cnnModel) ;
        urlwrite(fullfile('http://maxwell.cs.umass.edu/mvcnn/models/', ...
            [opts.cnnModel '.mat']), netFilePath) ;
        fprintf(' done!\n');
    end
    cnn = load(netFilePath);
end

% locate metric file
if isempty(opts.metricModel) && opts.applyMetric, 
    opts.applyMetric = false; 
    warning('No metric file specified. Post-processing turned off');
end
if opts.applyMetric
    metricFilePath = fullfile('data','models',[opts.metricModel '.mat']);
    if ~exist(metricFilePath,'file'),
        fprintf('Downloading model (%s) ...', opts.metricModel) ;
        urlwrite(fullfile('http://maxwell.cs.umass.edu/mvcnn/models/', ...
            [opts.metricModel '.mat']), metricFilePath) ;
        fprintf(' done!\n');
    end
    modelDimRedFV = load(metricFilePath);
end

% see if it's a multivew net
viewpoolIdx = find(cellfun(@(x)strcmp(x.name, 'viewpool'), cnn.layers));
if ~isempty(viewpoolIdx),
    if numel(viewpoolIdx)>1,
        error('More than one viewpool layers found!');
    end
    if ~isfield(cnn.layers{viewpoolIdx},'vstride'), 
        num_views = cnn.layers{viewpoolIdx}.stride; % old format
    else
        num_views = cnn.layers{viewpoolIdx}.vstride;
    end
    fprintf('CNN model is based on %d views. Will process %d views per mesh.\n', num_views, num_views);
else
    error('Computing a descriptor per shape requires a multi-view CNN.');
end

% work on mesh (or meshes)
if strcmpi((path_to_shape(end-2:end)),'off') || strcmpi((path_to_shape(end-2:end)),'obj')
    mesh_filenames(1).name = path_to_shape;
else
    mesh_filenames  = [dir( fullfile(path_to_shape, '*.off' ) ); dir( fullfile(path_to_shape, '*.obj') )];
    for i=1:length(mesh_filenames)  
        mesh_filenames(i).name = fullfile(path_to_shape,mesh_filenames(i).name);
    end
    if isempty(mesh_filenames)
        error('No obj/off meshes found in the specified folder!');
    end
end

descr = cell( 1, length(mesh_filenames));
fig = figure('Visible','off');
for i=1:length(mesh_filenames)
    fprintf('Loading shape %s ...', mesh_filenames(i).name);
    mesh = loadMesh( mesh_filenames(i).name );
    if isempty(mesh.F)
        error('Could not load mesh from file');
    else
        fprintf(' done!\n');
    end
    fprintf('Rendering shape %s ...', mesh_filenames(i).name);
    if num_views == 12
        ims = render_views(mesh, 'figHandle', fig);
    else
        ims = render_views(mesh, 'use_dodecahedron_views', true, 'figHandle', fig);
    end
    fprintf(' done!\n');
    outs = cnn_shape_get_features(ims, cnn, {opts.feature}, 'gpus', opts.gpus);
    out = outs.(opts.feature)(:);
    if opts.applyMetric
        out = single(modelDimRedFV.W*out);
    end
    descr{i} = out;
    out = double(out);  
    save( sprintf('%s_descriptor.txt', mesh_filenames(i).name(1:end-4)), 'out', '-ascii');
end
close(fig);
