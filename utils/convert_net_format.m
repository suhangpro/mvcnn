function net = convert_net_format(net, to_type)

if strcmpi(to_type,'old'), 
    for l = 1:numel(net.layers), 
        if isfield(net.layers{l},'weights'), 
            net.layers{l}.filters = net.layers{l}.weights{1};
            net.layers{l}.biases = net.layers{l}.weights{2};
            net.layers{l} = rmfield(net.layers{l},'weights');
        end
    end
elseif strcmpi(to_type,'new'), 
    for l=1:numel(net.layers), 
        if isfield(net.layers{l},'filters'), 
            net.layers{l}.weights = {net.layers{l}.filters, ...
                net.layers{l}.biases};
            net.layers{l} = rmfield(net.layers{l}, ...
                {'filters','biases'}); 
        end
    end
end