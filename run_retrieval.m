function [results] = run_retrieval(feat, imdb, varargin)
% e.g.:  r = run_retrieval('prob.mat','shapenet55v2','savePath','prob', ...
%                          'distFn',@(x1,x2) par_alldist(x1,[],'numWorkers',36,'maxParts',500));


opts.distFn = @(x1,x2) par_alldist(x1,x2,'numWorkers',12,'maxParts',100);
opts.topK = 1000;
opts.sets = {'train', 'val', 'test'}; 
opts.savePath = []; 
opts.saveDist = true; 
opts.resultType = 'fixedLength'; % 'fixedLength' | 'sameClass'
opts = vl_argparse(opts, varargin); 

if ischar(feat), 
  feat = load(feat);
  feat = feat.feat;
end

if ischar(imdb), 
  imdb = get_imdb(imdb);
end

nViews = numel(imdb.images.name) / size(feat,1); 

results = cell(2,numel(opts.sets)); 
for i = 1:numel(opts.sets), 
  setId = find(cellfun(@(s) strcmp(opts.sets{i},s),imdb.meta.sets));
  f = feat(imdb.images.set(1:nViews:end)==setId,:);
  sid = imdb.images.sid(imdb.images.set==setId);
  sid = sid(1:nViews:end); 
  nShapes = size(f,1);

  D = opts.distFn(f',f');
  if strcmpi(opts.resultType,'sameClass'), 
    [~,I] = max(f,[],2);
    sameLabelMask = arrayfun(@(l) (I'==l), I,'UniformOutput', false);
    results{1,i} = cellfun(@(c) sid(c), sameLabelMask, 'UniformOutput', false);
    results{2,i} = cell(nShapes,1);
    for j=1:nShapes, 
      [results{2,i}{j},I] = sort(D(j,sameLabelMask{j}),'ascend');
      topK = min(opts.topK, numel(I));
      results{2,i}{j} = results{2,i}{j}(1:topK);
      results{1,i}{j} = results{1,i}{j}(I(1:topK));
    end
  elseif strcmpi(opts.resultType,'fixedLength')
    [Y,I] = sort(D,2,'ascend');
    topK = min(opts.topK, numel(sid));
    I = I(:,1:topK); 
    dist_mat = Y(:,1:topK);
    result_mat = sid(I); 
    results{1,i} = cell(nShapes, 1);
    results{2,i} = cell(nShapes, 1); 	
    for j=1:nShapes, 
      results{1,i}{j} = result_mat(j,:);
      results{2,i}{j} = dist_mat(j,:); 
    end
  else
    error('Unknown option: %s', opts.resultType);
  end
  
  % write to file 
  if ~isempty(opts.savePath), 
    fprintf('Saving retrieval results to %s ...', fullfile(opts.savePath,opts.sets{i}));
    vl_xmkdir(fullfile(opts.savePath,opts.sets{i})); 
    for k=1:numel(sid), 
      fid = fopen(fullfile(opts.savePath,opts.sets{i},sprintf('%06d',sid(k))),'w+');
      r = results{1,i}{k};
      if opts.saveDist, 
        r = [r ; results{2,i}{k}]; 
        fprintf(fid,'%06d %f\n',r);
      else
        fprintf(fid,'%06d\n',r);
      end
      fclose(fid);
    end
    fprintf(' done!\n'); 
  end
  
end
