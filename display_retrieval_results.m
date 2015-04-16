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

angels = [0:30:330];
nViews = 6;
nRefViews = 12;
sketchDir = 'data/sketch160r6';
zedgeDir = 'data/modelnet40zedge';
toonDir = 'data/modelnet40toon';
queryView = 2;
dispViews = [1];% [1 5 9];
nTops = 8;

[nQuery, nRef] = size(r.dists);
[~,order] = sort(r.dists,2,'ascend');
dists0 = reshape(r.dists0,[nViews nQuery*nRefViews*nRef]);
[~,dispViewD] = min(reshape(0.5*(mean(dists0)+min(dists0)),[nQuery nRefViews nRef]),[],2);
dispViewD = squeeze(dispViewD);

N_W = 1 + nTops;
N_H = 2; % numel(dispViews);

q_sid = imdb.images.sid(imdb.images.set==4);
r_sid = imdb.images.sid(imdb.images.set==8);
q_sid = q_sid(1:nViews:end);
r_sid = r_sid(1:nRefViews:end);

figure;

[~,I] = sort(info.ap,'descend');
% I = randperm(length(info.ap));

for i=I(:)', 
    idx = find(imdb.images.sid==q_sid(i));
    q_im = imread(fullfile(sketchDir,imdb.images.name{idx(queryView)}));
    q_im = 255 - imdilate(255-q_im,strel('disk',3));
    vl_tightsubplot(N_H,N_W,1);
    imshow(q_im);
    %{-
    vl_tightsubplot(N_H,N_W,N_W+1);
    plot(info.recall(i,:),info.precision(i,:));
    axis square on;
    grid on;
    %}
    for j=1:nTops, 
        idx = find(imdb.images.sid==r.rankings(i,j));
        % for v=1:numel(dispViews), 
        r_im = imread(fullfile(zedgeDir,imdb.images.name{idx(dispViewD(i,order(i,j)))})); r_im = 255 - imdilate(255-r_im,strel('disk',3));
        vl_tightsubplot(N_H,N_W,1+j);
        imshow(r_im);
        r_im = imread(fullfile(toonDir,strrep(imdb.images.name{idx(dispViewD(i,order(i,j)))},'zedge','toon'))); % r_im = imfilter(r_im,fspecial('gaussian',[5 5],2));
        vl_tightsubplot(N_H,N_W,(1+nTops)+1+j);
        imshow(r_im);
        % end
    end
    waitforbuttonpress;
end
