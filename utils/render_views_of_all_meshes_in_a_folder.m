function render_views_of_all_meshes_in_a_folder(folder, varargin)
% calls 'render_views' for every shape found in the given folder

opts.ext = '.jpg';          % extension of target files
opts.range = [];            % if empty, all found shapes will be rendered, while a range [X:Y] will render shapes in the given range
opts.useUprightAssumption = true; % if true, 12 views will be used to render meshes, otherwise 80 views based on a dodecahedron
opts = vl_argparse(opts, varargin);

mesh_filenames = [rdir( sprintf('%s\\**\\*.obj', folder) ); rdir( sprintf('%s\\**\\**.off', folder) )];

if isempty( opts.range )
    range = 1:length( mesh_filenames );
else
    range = opts.range;
end
    
fig = figure('Visible','off');
for fi=range
    fprintf('Loading and rendering input shape %s...', mesh_filenames(fi).name );
    mesh = loadMesh( mesh_filenames(fi).name );
    if isempty(mesh.F)
        error('Could not load mesh from file');
    else
        fprintf('Done.\n');
    end
    if opts.useUprightAssumption
        ims = render_views(mesh, 'figHandle', fig);
    else
        ims = render_views(mesh, 'use_dodecahedron_views', true, 'figHandle', fig);
    end
    
    for ij=1:length(ims)
        imwrite( ims{ij}, sprintf('%s_%03d%s', mesh_filenames(fi).name(1:end-4), ij, opts.ext) );
    end
end
close(fig);
