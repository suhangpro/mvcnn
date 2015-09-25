function visualize_saliency( imdb, net, c )

[imdb.images.id, I] = sort(imdb.images.id);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);
imdb.images.sid = imdb.images.sid(I);

[imdb.images.sid, I] = sort(imdb.images.sid);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);
imdb.images.id = imdb.images.id(I);

nImages = numel(imdb.images.name);
nShapes = length(unique(imdb.images.sid));
nViews = nImages / nShapes;
sids = imdb.images.sid(1:nViews:end);
gts = imdb.images.class(1:nViews:end);


toonDir = 'data/modelnet40view';

% figure;
mag = zeros(1,nViews);
saliency = cell(1,nViews);
for i=randperm(length(sids)), 
    if ~strcmp(imdb.meta.classes{imdb.images.class((i-1)*nViews+1)},c);
        continue;
    end
    names = cellfun(@(s) fullfile(imdb.imageDir,s), ...
        imdb.images.name(imdb.images.sid==sids(i)),'UniformOutput',false);
    ims = get_image_batch(names,net.normalization);
    dzdy = zeros(1,1,length(imdb.meta.classes));
    dzdy(gts(i)) = 1;
    res = vl_simplenn(net,ims,dzdy);
    [~,I] = max(res(end).x);
    if imdb.images.class((i-1)*nViews+1)~=I,
        fprintf('%s: %s (passed)\n',imdb.meta.classes{imdb.images.class((i-1)*nViews+1)}, ...
            imdb.images.name{(i-1)*nViews+1});
        continue;
    end
    max_v = 0;
    min_v = Inf;
    for v=1:nViews, 
        vl_tightsubplot(2,nViews,v);
        imshow(imread(fullfile(imdb.imageDir,imdb.images.name{(i-1)*nViews+v})));
        % imshow(imread(fullfile(toonDir,strrep(imdb.images.name{(i-1)*nViews+v},'toon','view'))));
        dzdx = res(1).dzdx(:,:,:,v);
        mag(v) = sqrt(sum(dzdx(:).^2));
        saliency{v} = max(abs(dzdx),[],3);
        if max(saliency{v}(:))>max_v, max_v = max(saliency{v}(:)); end
        if min(saliency{v}(:))<min_v, min_v = min(saliency{v}(:)); end
    end
    [~,I] = sort(mag,'descend');
    for v=1:nViews, 
        saliency_map = (saliency{v}-min_v)/(max_v-min_v);
        vl_tightsubplot(2,nViews,nViews+v);
        if ismember(v,I(1:3)), 
            imshow(1-min(highlight_im(saliency_map.*2,[1.0 1.0 0 ],0.05),1));
            title(sprintf('%.2f',mag(v)*1e6),'color','blue');
        else
        	imshow(1-min(saliency_map.*2,1));
            title(sprintf('%.2f',mag(v)*1e6),'color','black');
        end
    end
    fprintf('%s: %s\n',imdb.meta.classes{imdb.images.class((i-1)*nViews+1)}, ...
        imdb.images.name{(i-1)*nViews+1});
    waitforbuttonpress;
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
