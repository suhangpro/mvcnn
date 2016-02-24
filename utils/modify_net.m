function net = modify_net(net, layer, varargin)

opts.mode = [];
opts.loc = [];
opts = vl_argparse(opts,varargin); 

switch opts.mode,
    case 'add_layer',
        if nargin<2 || isempty(layer), 
            error('Please provide the layer to be inserted!');
        end
        if isempty(opts.loc), 
            I = 0;
        else
            I = cellfun(@(x)(strcmp(x.name, opts.loc)), net.layers);
            I = find(I);
            if numel(I)~=1, 
                error('Ambiguous location: more than one %s layer!', opts.loc); 
            end;
        end
        net.layers = horzcat(net.layers(1:I), ...
                                layer, ...
                                net.layers(I+1:end));
    case 'rm_layer',
        if isempty(opts.loc), 
            error('Please indicate a layer to be removed!'); 
        end
        I = cellfun(@(x)(strcmp(x.name, opts.loc)), net.layers);
        I = find(I);
        if isempty(I), 
            error('No %s layer found!', opts.loc);
        elseif numel(I)~=1,
            error('Ambiguous location: more than one %s layer!', opts.loc);
        end;
        net.layers(I) = [];
    otherwise,
        fprintf('Empty/unknown action specified: %s\n', opts.mode);
end
