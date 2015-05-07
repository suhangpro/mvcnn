function [mesh labels labelsmap] = loadMesh(filestr, labels, labelsmap)
if nargin == 1
    labels = {};
    labelsmap = {};
end
if nargin == 2
    labelsmap = {};
end
    
% fprintf(1, '\nReading %s..\n', filestr);
file = fopen( strtrim( filestr ), 'rt');
if file == -1
    warning(['Could not open mesh file: ' filestr]);
    mesh = [];
    return;
end
mesh.filename = strtrim( filestr );

if strcmp( filestr(end-3:end), '.off')
    fgetl(file);
    line = strtrim(fgetl(file));
    [token,line] = strtok(line);
    numverts = eval(token);
    [token,line] = strtok(line);
    numfaces = eval(token);
    curvert = 0;
    curface = 0;
    mesh.V = zeros( 3, numverts, 'single' );
    mesh.F = zeros( 3, numfaces, 'single' );
    
    DATA = dlmread(filestr, ' ', 2, 0);
    DATA = DATA(1:numverts+numfaces, :);
    mesh.V(1:3, 1:numverts) = DATA(1:numverts, 1:3)';
    mesh.F(1:3, 1:numfaces) = DATA(numverts+1:numverts+numfaces, 2:4)' + 1;
elseif strcmp( filestr(end-3:end), '.obj')
    mesh.V = zeros(3, 10^6, 'single');
    mesh.Nv = zeros(3, 10^6, 'single');
    mesh.F = zeros(3, 5*10^6, 'uint32');
    mesh.parts = struct;
    mesh.faceLabels = zeros(1, 5*10^6, 'uint32');
    v = 0;
    f = 0;
    p = 0;
    vn = 0;
    cur_label_id = 0;
    cur_f_in_part = 0;
    
    while(~feof(file))
        line_type = fscanf(file,'%c',2);
        switch line_type(1)
            case '#'
                line = fgets(file);
            case 'v'
                if line_type(2) == 'n'
                    vn = vn + 1;
                    normal  = fscanf(file, '%f%f%f');
                    mesh.Nv(:, vn) = normal;
                elseif isspace( line_type(2) )
                    v = v + 1;
                    point = fscanf(file, '%f%f%f');
                    mesh.V(:, v) = point;
                else
                    fgets(file);
                end
            case 'f'
                f = f + 1;
                face = fscanf(file, '%u%u%u');
                mesh.F(:, f) = face;
%                 cur_f_in_part = cur_f_in_part + 1;
%                 mesh.parts(p).faces(cur_f_in_part) = f;
%                 mesh.faceLabels(f) = cur_label_id;
%             case 'g'
%                 if p ~= 0
%                     mesh.parts(p).faces = mesh.parts(p).faces( 1:cur_f_in_part );
%                 end
%                 p = p + 1;
%                 label = fgetl(file);                
%                 mesh.parts(p).name = label;
%                 label( (label >= '0' & label <= '9') | label == '_' | label == '-' ) = [];
%                 label = lower(label);
%                 disp(['Read label: ' mesh.parts(p).name]);
%                 [labels, cur_label_id] = searchLabels( labels, label );
%                 mesh.parts(p).faces = zeros(1, 5*10^6, 'uint32');
%                 cur_f_in_part = 0;
%                 if length( labelsmap ) < cur_label_id
%                     labelsmap{cur_label_id} = {};
%                 end
%                 labelsmap{cur_label_id} = [ labelsmap{cur_label_id} {mesh.filename} ];
            otherwise        
                if isspace(line_type(1))
                    fseek(file, -1, 'cof');
                    continue;
                end
                fprintf('last string read: %c%c %s', line_type(1), line_type(2), fgets(file));                
                fclose(file);                
                error('only triangular obj meshes are supported with vertices, normals, groups, and vertex normals.');
        end
    end
    if p ~= 0
        mesh.parts(p).faces = mesh.parts(p).faces( 1:cur_f_in_part );
    end
    
    mesh.V = mesh.V(:, 1:v);
    mesh.F = mesh.F(:, 1:f);
    mesh.Nv = mesh.Nv(:, 1:v);
    mesh.faceLabels  = mesh.faceLabels(1:f);
    
    nullfaces = find( mesh.faceLabels == 0);
    if ~isempty( nullfaces )
%        warning('faces with no label found; this can be very bad!');
        mesh.parts(p+1).name = '__null__';
        mesh.parts(p+1).labelid = 0;
        mesh.parts(p+1).faces = nullfaces;
    end
end

fclose(file);

end



function [labels,pos] = searchLabels( labels, currentlabel )
pos = strmatch(currentlabel, labels, 'exact');
if isempty( pos )
    pos = strmatch(currentlabel(1:end-1), labels, 'exact');
    if isempty( pos )
        labels{end+1} = currentlabel;
        pos = length(labels);
    else
        return;
    end
else
    return;
end
end


%%%%%% old obj file reading
%     lines = cell(10^7, 1);
%     i = 0;
%     fprintf(1, '%.2f%% complete\n', 0.0);
%     while ~feof(file)
%         i = i + 1;
%         lines{i} = fgets(file, 1024);
%     end
%     lines = lines(1:i);
%
%     mesh.V = ones(3, length(lines), 'single');
%     mesh.Nv = zeros(3, length(lines), 'single');
%     mesh.F = zeros(3, length(lines), 'uint32');
%     v = 0;
%     f = 0;
%     vn = 0;
%
%     for i=1:length(lines)
%         if mod(i, 3000) == 0
%             fprintf(1, '%.2f%% complete\n', 100 * (i/length(lines)));
%         end
%         line = lines{i};
%         if line(1) == 'v' && line(2) == 'n'
%             line = line(3:end);
%             vn = vn + 1;
%             [mesh.Nv(1, vn), line] = fnumtok(line);
%             [mesh.Nv(2, vn), line] = fnumtok(line);
%             mesh.Nv(3, vn) = fnumtokNR(line);
%         elseif line(1) == 'v'
%             line = line(3:end);
%             v = v + 1;
%             [mesh.V(1, v), line] = fnumtok(line);
%             [mesh.V(2, v), line] = fnumtok(line);
%             mesh.V(3, v) = fnumtokNR(line);
%         elseif line(1) == 'f'
%             line = line(3:end);
%             f = f + 1;
%             [token line] = strtok(line);
%             mesh.F(1, f) = numtokNR(token);
%             [token line] = strtok(line);
%             mesh.F(2, f) = numtokNR(token);
%             token = strtokNR(line);
%             mesh.F(3, f) = numtokNR(token);
%         end
%     end
%
%     mesh.V = mesh.V(:, 1:v);
%     mesh.F = mesh.F(:, 1:f);
%     mesh.Nv = mesh.Nv(:, 1:v);
%     fprintf(1, '%.2f%% complete\n', 100.0);
% end

%
%
%
% function [token line] = strtok(line)
% [token pos] = textscan(line, '%s', 1);
% token = token{1}{1};
% line = line(pos+1:end);
%
% function [token line] = fnumtok(line)
% [token pos] = textscan(line, '%f', 1);
% token = token{1};
% line = line(pos+1:end);
%
% function [token line] = numtok(line)
% [token pos] = textscan(line, '%d', 1);
% token = token{1};
% line = line(pos+1:end);
%
%
% function token = strtokNR(line)
% token = textscan(line, '%s', 1);
% token = token{1}{1};
%
% function token = fnumtokNR(line)
% token = textscan(line, '%f', 1);
% token = token{1};
%
% function token = numtokNR(line)
% token = textscan(line, '%d', 1);
% token = token{1};
%
