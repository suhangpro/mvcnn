function display_retrieval_results()

% r.rankings 
% r.dists
% r.dists0 
% info 
% imdb 
if ~exist('r','var'), load('data/eval.mat'); end;

% sort imdb.images wrt id
[imdb.images.id, I] = sort(imdb.images.id);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);
if isfield(imdb.images,'sid'), imdb.images.sid = imdb.images.sid(I); end

% sort imdb.images wrt sid
if isfield(imdb.images,'sid'),
    [imdb.images.sid, I] = sort(imdb.images.sid);
    imdb.images.name = imdb.images.name(I);
    imdb.images.class = imdb.images.class(I);
    imdb.images.set = imdb.images.set(I);
    imdb.images.id = imdb.images.id(I);
end

nViews = 6;
nRefViews = 12;
sketchDir = 'data/sketch160r6';
zedgeDir = 'data/modelnet40zedge';
toonDir = 'data/modelnet40toon';
viewDir = 'data/modelnet40view';
saveDir = 'data/retrieval-results';vl_xmkdir(saveDir);
queryView = 2;
dispViews = [1];% [1 5 9];
nTops = 10;
border = 0.06;

[nQuery, nRef] = size(r.dists);
[~,order] = sort(r.dists,2,'ascend');
dists0 = reshape(r.dists0,[nViews nQuery*nRefViews*nRef]);
[~,dispViewD] = min(reshape(0.5*(mean(dists0)+min(dists0)),[nQuery nRefViews nRef]),[],2);
dispViewD = squeeze(dispViewD);

N_W = 1 + nTops;
N_H = 10; %5; %2; % numel(dispViews);

q_sid = imdb.images.sid(imdb.images.set==4);
q_sid = q_sid(1:nViews:end);
q_class = imdb.images.class(imdb.images.set==4);
q_class = q_class(1:nViews:end);

figure;

[~,I] = sort(info.ap,'descend');
% I = randperm(length(info.ap));

I = [18 28 47 71 95 152 170 176 246 115]';
classes = [1 2 3 5 7 8 9 11 12 13];

% plotted = [];
k=1;
for i=I(:)', 
    % k = numel(plotted)+1;
    gt = q_class(i);
    % if ismember(gt,plotted), continue; end;
    % plotted(end+1) = gt;
    idx = find(imdb.images.sid==q_sid(i));
    q_im = imread(fullfile(sketchDir,imdb.images.name{idx(queryView)}));
    q_im = 255 - imdilate(255-q_im,strel('disk',3));
    vl_tightsubplot(N_H,N_W,N_W*(k-1)+1);
    imshow(q_im);
    %{
    vl_tightsubplot(N_H,N_W,N_W+1);
    plot(info.recall(i,:),info.precision(i,:));
    axis square on;
    grid on;
    %}
    % [~,v] = max(histc(dispViewD(i,order(i,1:nTops)),1:12));
    skipped = -1;
    for j=1:nTops, 
        c = -1;
        while ~ismember(c,classes), 
            skipped = skipped + 1;
            v = dispViewD(i,order(i,j+skipped));
            idx = find(imdb.images.sid==r.rankings(i,j+skipped));
            c = imdb.images.class(idx(v));
        end
        % for v=1:numel(dispViews), 
        % r_im = imread(fullfile(zedgeDir,imdb.images.name{idx(v)})); r_im = 255 - imdilate(255-r_im,strel('disk',3));
        r_im = imread(fullfile(viewDir,strrep(imdb.images.name{idx(v)},'zedge','view'))); % r_im = imfilter(r_im,fspecial('gaussian',[5 5],2));
        if gt==c, 
            % r_im = highlight_im(r_im,[0 1 0],border);
        else
            r_im = highlight_im(r_im,[1 0 0],border);
        end
        vl_tightsubplot(N_H,N_W,N_W*(k-1)+1+j);
        imshow(r_im);
        % end
    end
    % print(fullfile(saveDir,[num2str(i) '.jpg']),'-djpeg');
    k = k+1;% if numel(plotted)==N_H, break; end
    % waitforbuttonpress;
end

end

function im = highlight_im(im0,c,w) 
    if isinteger(im0), im0 = double(im0)/255; end
    if isinteger(c), c = double(c)/255; end
    if size(im0,3)==1, im0 = repmat(im0,[1 1 3]); end
    sz = size(im0); 
    if w<1, w = floor(mean(sz(1:2))*w); end
    im = bsxfun(@times,ones(size(im0)),reshape(c,[1 1 3]));
    im(w+1:end-w,w+1:end-w,:) = im0(w+1:end-w,w+1:end-w,:);
end
