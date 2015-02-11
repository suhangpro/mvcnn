function subWins = get_augmentation_matrix( augmentationType, varargin )
%GET_AUGMENTATION Build augmentation matrix 
% 
%   augmentationType:: 'nr3'
%       1st field(f|n) indicates whether include flipped copy or not
%       2nd field(s|r) indicates type of region - Square or Rectangle
%       3rd field(1..4) indicates number of levels 
%       note: 'none', 'ns1', 'nr1' are equivalent
%   `width`:: [1, 0.75, 0.5, 1/3]
%       width of windows in each layer

opts.width = [1, 0.75, 0.5, 1/3];
opts = vl_argparse(opts,varargin);
width = opts.width;

if strcmp(augmentationType,'none'), 
    augmentationType = 'ns1';
end
nLevels = str2double(augmentationType(3));
if nLevels>length(width), 
    error('Too many levels.');
end
if augmentationType(2)=='s', 
    subWins = [0;1;0;1];
    for l=2:nLevels, 
        sv = linspace(0,1-width(l),l);
        [XX,YY] = meshgrid(sv,sv);
        sxy = reshape(permute(cat(3,XX,YY),[3,1,2]),[2,l^2]);
        subWins = [subWins [sxy(1,:) ; width(l)*ones(1,l^2) ;sxy(2,:); ...
            width(l)*ones(1,l^2)]];
    end
elseif augmentationType(2)=='r', 
    sv = [];
    w = [];
    for l=1:nLevels, 
        sv = [sv linspace(0,1-width(l),l)];
        w = [w width(l)*ones(1,l)];
    end
    [XX_sv,YY_sv] = meshgrid(sv,sv);
    [XX_w,YY_w] = meshgrid(w,w);
    sxy = reshape(permute(cat(3,XX_sv,YY_sv),[3,1,2]),[2,numel(XX_sv)]);
    wxy = reshape(permute(cat(3,XX_w,YY_w),[3,1,2]),[2,numel(XX_w)]);
    subWins = [sxy(1,:) ; wxy(1,:) ; sxy(2,:) ; wxy(2,:)];
else
    error('Unknow augmentation type: %s', augmentationType);
end
subWins(end+1,:) = 0;
if augmentationType(1)=='f', 
    subWins = [subWins subWins];
    subWins(end,size(subWins,2)/2+1:end) = 1;
end

end

