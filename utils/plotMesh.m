function plotMesh(mesh, style, az, el)

% mesh.F = mesh.f;
% mesh.V = mesh.v;
% plotMesh(mesh)
% plotMesh(mesh, style)
%
% style     {figure} | mesh | solid | solidbw

if nargin < 2
    style = 'figure';
end
    
if strcmpi(style, 'figure')
    
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'none', ...
                'EdgeColor', 'k','EdgeLighting','flat','AmbientStrength',.4);

    set(gcf, 'Renderer', 'OpenGL');

    axis equal;
    axis tight;
    grid on;
    if nargin >= 4 view(az,el); else view(3); end

elseif strcmpi(style, 'mesh')
    
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'none', 'EdgeColor', 'w', ...
        'AmbientStrength', 0.4, 'EdgeLighting', 'flat');
  
    set(gcf, 'Color', 'k', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');

    axis equal;
    axis off;
    if nargin >= 4 view(az,el); else view(3); end
    % camlight;

elseif strcmpi(style, 'solid')
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'none', ...
        'AmbientStrength', 0.2, 'FaceLighting', 'phong', 'SpecularStrength', 1.0, 'SpecularExponent', 100);
    
    if isfield(mesh, 'Nv')
        set(h, 'VertexNormals', -mesh.Nv(1:3,:)'); 
    end
    
    if isfield(mesh, 'C')
        set(h, 'FaceVertexCData', mesh.C);
        set(h, 'FaceColor', 'flat');
    end
    
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');
    
    axis equal;    
    axis off;
    if nargin >= 4 view(az,el); else view(3); end
    camlight;
    
elseif strcmpi(style, 'solidbw')
    
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'k', ...
        'AmbientStrength', 0.7, 'FaceLighting', 'flat', 'EdgeLighting', 'none');
        
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');

    axis equal;    
    axis off;
    if nargin >= 4 view(az,el); else view(3); end
    % camlight;

elseif strcmpi(style, 'solidbws')
    
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'k', ...
        'AmbientStrength', 0.7, 'FaceLighting', 'phong', 'EdgeLighting', 'none');
        
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');

    axis equal;    
    axis off;
    if nargin >= 4 view(az,el); else view(3); end
    % camlight;
    
elseif strcmpi(style, 'ghost')
    
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'k', ...
        'AmbientStrength', 0.4, 'FaceLighting', 'phong', 'EdgeLighting', 'none', 'FaceAlpha',.5);
        
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');

    axis equal;    
    axis off;
    if nargin >= 4 view(az,el); else view(3); end
    % camlight;
    
elseif strcmpi(style, 'ghosts')
    h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'none', ...
        'AmbientStrength', 0.2, 'FaceLighting', 'phong', 'SpecularStrength', 1.0, 'SpecularExponent', 100, 'FaceAlpha',.5);
    
    if isfield(mesh, 'Nv')
        set(h, 'VertexNormals', -mesh.Nv(1:3,:)'); 
    end
    
    if isfield(mesh, 'C')
        set(h, 'FaceVertexCData', mesh.C);
        set(h, 'FaceColor', 'interp');
    end
    
    set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    set(gca, 'Projection', 'perspective');
    
    axis equal;    
    axis off;
    if nargin >= 4 view(az,el); else view(3); end
    camlight;
end



