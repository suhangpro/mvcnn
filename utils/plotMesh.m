function h = plotMesh(mesh, style, az, el)

if nargin < 2
    style = 'solid';
end

if strcmpi(style, 'mesh')    
    h = trimesh(mesh.F', mesh.V(1,:)', mesh.V(2,:)', mesh.V(3,:)', 'FaceColor', 'none', 'EdgeColor', 'w', ...
        'AmbientStrength', 0.4, 'EdgeLighting', 'flat');    
    set(gcf, 'Color', 'k', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');    
    axis equal;
    axis off;
    view(az,el);    
elseif strcmpi(style, 'solid')
    h = trimesh(mesh.F', mesh.V(1,:)', mesh.V(2,:)' ,mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'none', ...
        'AmbientStrength', 0.3, 'DiffuseStrength', 0.6, 'SpecularStrength', 0.0, 'FaceLighting', 'flat');
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');    
    axis equal;
    axis off;
    view(az,el);
    camlight('HEADLIGHT');    
elseif strcmpi(style, 'solidphong')
    mesh = normals(mesh);
    h = trimesh(mesh.F', mesh.V(1,:)', mesh.V(2,:)' ,mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'none', ...
        'AmbientStrength', 0.3, 'DiffuseStrength', 0.6, 'SpecularStrength', 0.0, 'FaceLighting', 'gouraud', ...
        'VertexNormals', -mesh.Nv(1:3,:)', 'BackFaceLighting', 'reverselit');
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');    
    axis equal;
    axis off;
    view(az,el);
    camlight('HEADLIGHT');    
elseif strcmpi(style, 'soliddoublesided')
    mesh = normals(mesh);    
    lx = cos(az) * cos(el);
    ly = cos(az) * sin(el);
    lz = sin(az);
    lightdir = [lx ly lz]';
    mesh.C = zeros( size(mesh.V, 2), 3 );
    for i=1:size(mesh.V, 2)
        mesh.C(i, :) = .3 + .6 * max( sum( lightdir .* mesh.Nv(:, i) ), sum( -lightdir .* mesh.Nv(:, i) ) );
    end  
    h = trimesh(mesh.F', mesh.V(1,:)', mesh.V(2,:)' ,mesh.V(3,:)', 'EdgeColor', 'none', ...
        'FaceVertexCData', mesh.C, 'FaceColor', 'interp' );
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');    
    axis equal;
    axis off;
    view(az,el);
    camlight;        
end

