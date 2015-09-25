function display_right_wrong(decTest,imdb)
rng(1);
[imdb.images.id,I] = sort(imdb.images.id);
imdb.images.class = imdb.images.class(I);
imdb.images.name = imdb.images.name(I);
imdb.images.set = imdb.images.set(I);
% Cs = imdb.meta.classes;
Cs = {
    'pen'
    'crab'
    'snail'
    'squirrel'
    'guitar'
    'butterfly'
    'church'
    'flying bird'
    'laptop'
    'strawberry'
    'knife'
    'bicycle'
    'grapes',
    'hammer'
    };
Cs = Cs(randperm(length(Cs)));
[~,CIs] = ismember(Cs,imdb.meta.classes);
nTops = [6 6];

[~,pred] = max(decTest,[],2);
gt = imdb.images.class(imdb.images.set==3)';
names = imdb.images.name(imdb.images.set==3);
trueMask = gt==pred;

saveDir = '~/Desktop/sketch_wrong';
vl_xmkdir(saveDir);

for c=1:numel(Cs), 
    fprintf('%10s ',Cs{c});
    inds1 = (pred==CIs(c)) & trueMask;
    inds0 = (pred==CIs(c)) & ~trueMask;
    [~,I]=sort(decTest(inds1,CIs(c)),'ascend');
    idxs = find(inds1);
    idxs = idxs(I);
    % idxs = idxs(randperm(length(idxs)));
    for i=1:nTops(1), 
        vl_tightsubplot(length(Cs),sum(nTops),i+(c-1)*sum(nTops));
        im0 = imread(fullfile(imdb.imageDir,names{idxs(i)}));
        im0 = imresize(im0,[400,400]);
        im0 = 1-im2double(im0);
        im0 = imfilter(im0,fspecial('disk',2)*15);
        % im = highlight_im(im,[0 1 0],0.05);
        im = ones(size(im0,1),size(im0,2),3);
        im(:,:,1) = im(:,:,1)-im0;
        im(:,:,3) = im(:,:,3)-im0;
        imshow(im);
        if i==1, text(25,50,imdb.meta.classes(gt(idxs(i))),'fontsize',18,'color','black'); end;
    end
    [~,I]=sort(decTest(inds0,CIs(c)),'descend');
    idxs = find(inds0); 
    idxs = idxs(I);
    for i=1:min(nTops(2),length(idxs)), 
        vl_tightsubplot(length(Cs),sum(nTops),nTops(1)+i+(c-1)*sum(nTops));
        im0 = imread(fullfile(imdb.imageDir,names{idxs(i)}));
        im0 = imresize(im0,[400 400]);
        im0 = 1-im2double(im0);
        im0 = imfilter(im0,fspecial('disk',2)*15);
        % im = highlight_im(im,[1 0 0],0.05);
        im = ones(size(im0,1),size(im0,2),3);
        im(:,:,2) = im(:,:,2)-im0;
        im(:,:,3) = im(:,:,3)-im0;
        imshow(im);
        text(25,50,imdb.meta.classes(gt(idxs(i))),'fontsize',18,'color','black');
    end
    for i=min(nTops(2),length(idxs))+1:nTops(2),
        vl_tightsubplot(length(Cs),sum(nTops),nTops(1)+i+(c-1)*sum(nTops));
        im = ones(10,10,3);
        imshow(im);
    end
    %{
    keydown = waitforbuttonpress;
    if keydown==1, 
        fprintf('GOOD\n');
    else
        fprintf('\n');
    end
    %}
    %{
    fprintf('\n');
    print(gcf,'-depsc',fullfile(saveDir,[strrep(Cs{c},' ','_') '.eps']));
    print(gcf,'-dpng',fullfile(saveDir,'png',[strrep(Cs{c},' ','_') '.png']));
    %}
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
