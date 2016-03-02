function [r1] = rerank_retrieval(r1, r2, imdb, savePath, sets)
%RERANK_RETRIEVAL re-rank results in r1 using distances in r2
% r1, r2 are results of run_retrieval()

nSets = size(r1,2);
if ~exist('sets','var') || isempty(sets), 
  sets = {'train', 'val', 'test'};
  sets = sets(1:nSets); 
end
if ~exist('savePath', 'var') || isempty(savePath), 
  savePath = [];
end

nViews = numel(imdb.images.sid) / numel(unique(imdb.images.sid)); 

if size(r1,1)>1, saveDist = true; end

for s = 1:nSets, 
  setId = find(cellfun(@(v) strcmp(sets{s},v),imdb.meta.sets)); 
  sid = imdb.images.sid(imdb.images.set==setId);   
  sid = sid(1:nViews:end); 
  for i = 1:numel(r1{1,s}),
    [Y,I]=ismember(r1{1,s}{i}, r2{1,s}{i});
    assert(all(Y));
    [~,I] = sort(I);
    r1{1,s}{i} = r1{1,s}{i}(I); 
    if saveDist, r1{2,s}{i} = r1{2,s}{i}(I); end
  end
  % write to file
  if ~isempty(savePath), 
    fprintf('Saving re-ranked results to %s ...', fullfile(savePath,sets{s}));
    vl_xmkdir(fullfile(savePath,sets{s}));
    for i=1:numel(sid)
      fid = fopen(fullfile(savePath,sets{s},sprintf('%06d',sid(i))),'w+');
      r = r1{1,s}{i};
      if saveDist, 
        r = [r ; r1{2,s}{i}];
        fprintf(fid, '%06d %f\n',r);
      else
        fprintf(fid, '%06d\n', r);
      end
      fclose(fid);
    end
    fprintf(' done!\n'); 
  end
end
