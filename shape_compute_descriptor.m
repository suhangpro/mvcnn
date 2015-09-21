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
%   `cnn_model`:: (default) 'cnn-modelnet40-v1.mat'
%       this is a matlab file with the saved CNN parameters
%       if the default file is not found, it will be downloaded from our
%       server.
%       Note: The default mat file assumes that shapes that are
%       upright oriented according to +Z axis!
%       if you want to use the CNN trained *without upright assumption*, use
%       'cnn-modelnet40-v2.mat'
%   `post_process_desriptor_metric`:: (default) true
%       set to false to disable transforming descriptor based on our
%       learned distance metric
%   `metric_model`:: (default:) 'metric-relu7-v1.mat'
%       this is a matlab file with the saved metric parameters
%       if the default file is not found, it will be downloaded from our
%       server
%       if you want to use the model trained *without upright assumption*, use
%       'metric-relu7-v2.mat'
%   `gpuMode`:: (default) false
%       set to true to compute on GPU
%   `numWorkers`:: (default) 12
%       number of CPU workers, only in use when gpuMode is false

addpath(genpath('utils'));
run dependencies/vlfeat/toolbox/vl_setup.m
run dependencies/matconvnet/matlab/vl_setupnn.m

if nargin<1 || isempty(path_to_shape),
    imdbName = 'data/';
end

% default options
opts.cnn_model = 'cnn-modelnet40-v1.mat';
opts.post_process_desriptor_metric = true;
opts.metric_model = 'metric-relu7-v1.mat';
opts.gpuMode = false;
opts.numWorkers = 12;
opts = vl_argparse(opts,varargin);

if exist(opts.cnn_model, 'file')
    fprintf('Loading CNN model from mat file...');
    cnn = load(opts.cnn_model);
    fprintf('Done.\n');
else
    fprintf('Downloading CNN model from our server...');
    url = ['http://maxwell.cs.umass.edu/deep-shape-data/models/' opts.cnn_model];
    websave(opts.cnn_model,url);
    if exist(opts.cnn_model, 'file')
        cnn = load(opts.cnn_model);
	cnn = convert_net_format(cnn,'old');
        fprintf('Done.\n');
    else
        error('Could not download mat file from our server. Please check internet connection or contact us.');
    end
end

if opts.post_process_desriptor_metric
    if exist(opts.metric_model, 'file')
        fprintf('Loading metric model from mat file...');
        modelDimRedFV = load(opts.metric_model);
        fprintf('Done.\n');
    else
        fprintf('Downloading metric model from our server...');
        url = ['http://maxwell.cs.umass.edu/deep-shape-data/models/' opts.metric_model];
        websave(opts.metric_model,url);
        if exist(opts.metric_model, 'file')
            modelDimRedFV = load(opts.metric_model);
            fprintf('Done.\n');
        else
            error('Could not download mat file from our server. Please check internet connection or contact us.');
        end
    end
end


% see if it's a multivew net
viewpoolIdx = find(cellfun(@(x)strcmp(x.name, 'viewpool'), cnn.layers));
if ~isempty(viewpoolIdx),
    if numel(viewpoolIdx)>1,
        error('More than one viewpool layers found!');
    end
    num_views = cnn.layers{viewpoolIdx}.stride;
    fprintf('CNN model is based on %d views. Will process %d views per mesh.\n', num_views, num_views);
else
    error('Computing a descriptor per shape requires a multi-view CNN.');
end

% work on mesh (or meshes)
if strcmpi((path_to_shape(end-2:end)),'off') || strcmpi((path_to_shape(end-2:end)),'obj')
    mesh_filenames(1).name = path_to_shape;
else
    mesh_filenames  = [dir( strcat(path_to_shape, '/*.off' ) ); dir( strcat(path_to_shape, '/*.obj') )];
    for i=1:length(mesh_filenames)  
        mesh_filenames(i).name = [path_to_shape '/' mesh_filenames(i).name];
    end
    if isempty(mesh_filenames)
        error('No obj/off meshes found in the specified folder!');
    end
end

descr = cell( 1, length(mesh_filenames));
fig = figure('Visible','off');
for i=1:length(mesh_filenames)
    fprintf('Loading input shape %s...', mesh_filenames(i).name);
    mesh = loadMesh( mesh_filenames(i).name );
    if isempty(mesh.F)
        error('Could not load mesh from file');
    else
        fprintf('Done.\n');
    end
    if num_views == 12
        ims = render_views(mesh, 'figHandle', fig);
    else
        ims = render_views(mesh, 'use_dodecahedron_views', true, 'figHandle', fig);
    end
    outs = get_cnn_activations(ims, cnn, [], {'relu7','prob'});
    out = outs.relu7(:);
    if opts.post_process_desriptor_metric
        out = double(modelDimRedFV.W*out);
    end
    descr{i} = out;
    save( sprintf('%s_descriptor.txt', mesh_filenames(i).name(1:end-4)), 'out', '-ascii');
end
close(fig);
