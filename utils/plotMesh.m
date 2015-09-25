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
        'AmbientStrength', 0.2, 'FaceLighting', 'phong', 'SpecularStrength', 0.0, 'SpecularExponent', 100);    
    if isfield(mesh, 'Nv')
        set(h, 'VertexNormals', -mesh.Nv(1:3,:)');
    end    
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');    
    axis equal;
    axis off;
    view(az,el);
    camlight;    
end