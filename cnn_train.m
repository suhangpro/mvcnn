function [net, info] = cnn_train(net, imdb, getBatch, varargin)
% CNN_TRAIN   modified from matconvnet/examples/cnn_train.m

opts.train = find(imdb.images.set==1) ;
opts.val = find(imdb.images.set==2) ;
opts.numEpochs = 300 ;
opts.multiview = false;
opts.batchSize = 256 ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9;
opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

if opts.multiview, 
    [~,I] = sort(imdb.images.sid(opts.train));
    opts.train = opts.train(I);
    [~,I] = sort(imdb.images.sid(opts.val));
    opts.val = opts.val(I);
    nInstances = length(unique(imdb.images.sid));
    nViews = length(imdb.images.id)/nInstances;
    if mod(opts.batchSize,nViews)~=0, 
        error('Batch size incompatible with #views');
    end
else
    nViews = 1;
end

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
  if isfield(net.layers{i},'weights'), 
    J = numel(net.layers{i}.weights);
    for j=1:J
      net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), ...
        class(net.layers{i}.weights{j})) ;
    end
    if ~isfield(net.layers{i}, 'learningRate')
      net.layers{i}.learningRate = ones(1,J,class(net.layers{i}.weights{j}));
    end
    if ~isfield(net.layers{i}, 'weightDecay')
      net.layers{i}.weightDecay = ones(1,J,class(net.layers{i}.weights{j}));
    end
  elseif isfield(net.layers{i},'filters') % old format 
    net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
      class(net.layers{i}.filters)) ;
    net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
      class(net.layers{i}.biases)) ;
    if ~isfield(net.layers{i}, 'filtersLearningRate')
      net.layers{i}.filtersLearningRate = 1 ;
    end
    if ~isfield(net.layers{i}, 'biasesLearningRate')
      net.layers{i}.biasesLearningRate = 1 ;
    end
    if ~isfield(net.layers{i}, 'filtersWeightDecay')
      net.layers{i}.filtersWeightDecay = 1 ;
    end
    if ~isfield(net.layers{i}, 'biasesWeightDecay')
      net.layers{i}.biasesWeightDecay = 1 ;
    end
  end
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for i=1:numel(net.layers)
    if isfield(net.layers{i},'weights'),
      J = numel(net.layers{i}.weights);
      for j=1:J
        net.layers{i}.momentum{j} = gpuArray(net.layers{i}.momentum{j}) ;
      end
    elseif isfield(net.layers{i},'filters'), % old format
      net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
      net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
    end
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;
info.val.speed = [] ;

lr = 0 ;
res = [] ;
for epoch=1:opts.numEpochs  
  prevLr = lr ;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to where we stopped
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file'), continue ; end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end

  info.train.objective(end+1) = 0 ;
  info.train.error(end+1) = 0 ;
  info.train.topFiveError(end+1) = 0 ;
  info.train.speed(end+1) = 0 ;
  info.val.objective(end+1) = 0 ;
  info.val.error(end+1) = 0 ;
  info.val.topFiveError(end+1) = 0 ;
  info.val.speed(end+1) = 0 ;

  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
    for l=1:numel(net.layers)
      if isfield(net.layers{l},'weights'),
        for j=1:numel(net.layers{l}.momentum),
          net.layers{l}.momentum{j} = 0 * net.layers{l}.momentum{j} ;
        end
      elseif isfield(net.layers{l},'filters'), % old format
        net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
        net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
      end
    end
  end

  trainRs = reshape(opts.train,[nViews numel(opts.train)/nViews]);
  I = randperm(numel(opts.train)/nViews);
  train = reshape(trainRs(:,I),[1 numel(opts.train)]);
  n_train = 0; 
  for t=1:opts.batchSize:numel(train)
    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    batch_time = tic ;
    fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    labels = labels(1:nViews:end);
    if opts.prefetch
      nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      im = gpuArray(im) ;
    end

    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, one, res, ...
      'conserveMemory', opts.conserveMemory, ...
      'sync', opts.sync) ;

    % gradient step
    for l=1:numel(net.layers)
      if isfield(net.layers{l},'weights'), 
        J = numel(net.layers{l}.weights);
        for j=1:J,
          net.layers{l}.momentum{j} = ...
            opts.momentum * net.layers{l}.momentum{j} ...
            - (lr * net.layers{l}.learningRate(j)) * ...
            (opts.weightDecay * net.layers{l}.weightDecay(j)) * net.layers{l}.weights{j} ...
            - (lr * net.layers{l}.learningRate(j)) / numel(labels) * res(l).dzdw{j} ;
          net.layers{l}.weights{j} = net.layers{l}.weights{j} + net.layers{l}.momentum{j};
        end
      elseif isfield(net.layers{l},'filters'), % old format
        net.layers{l}.filtersMomentum = ...
          opts.momentum * net.layers{l}.filtersMomentum ...
          - (lr * net.layers{l}.filtersLearningRate) * ...
          (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
          - (lr * net.layers{l}.filtersLearningRate) / numel(labels) * res(l).dzdw{1} ;
        
        net.layers{l}.biasesMomentum = ...
          opts.momentum * net.layers{l}.biasesMomentum ...
          - (lr * net.layers{l}.biasesLearningRate) * ....
          (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
          - (lr * net.layers{l}.biasesLearningRate) / numel(labels) * res(l).dzdw{2} ;
        
        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
      end
    end

    % print information
    batch_time = toc(batch_time) ;
    speed = numel(batch)/batch_time ;
    info.train = updateError(opts, info.train, net, res, batch_time) ;

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    n_train = n_train + numel(labels) ;
    fprintf(' err %.1f err5 %.1f', ...
      info.train.error(end)/n_train*100, info.train.topFiveError(end)/n_train*100) ;
    fprintf('\n') ;

    % debug info
    if opts.plotDiagnostics
      figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
    end
  end % next batch

  % evaluation on validation set
  val = opts.val;
  n_val = 0;
  for t=1:opts.batchSize:numel(val)
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    labels = labels(1:nViews:end);
    if opts.prefetch
      nextBatch = val(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(val))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      im = gpuArray(im) ;
    end

    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, [], res, ...
      'disableDropout', true, ...
      'conserveMemory', opts.conserveMemory, ...
      'sync', opts.sync) ;

    % print information
    batch_time = toc(batch_time) ;
    speed = numel(batch)/batch_time ;
    info.val = updateError(opts, info.val, net, res, batch_time) ;

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    n_val = n_val + numel(labels) ;
    fprintf(' err %.1f err5 %.1f', ...
      info.val.error(end)/n_val*100, info.val.topFiveError(end)/n_val*100) ;
    fprintf('\n') ;
  end

  % save
  info.train.objective(end) = info.train.objective(end) / n_train ;
  info.train.error(end) = info.train.error(end) / n_train  ;
  info.train.topFiveError(end) = info.train.topFiveError(end) / n_train ;
  info.train.speed(end) = numel(train) / info.train.speed(end) ;
  info.val.objective(end) = info.val.objective(end) / n_val ;
  info.val.error(end) = info.val.error(end) / n_val ;
  info.val.topFiveError(end) = info.val.topFiveError(end) / n_val ;
  info.val.speed(end) = numel(val) / info.val.speed(end) ;
  save(modelPath(epoch), 'net', 'info') ;

  figure(1) ; clf ;
  subplot(1,2,1) ;
  semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
  semilogy(1:epoch, info.val.objective, 'b') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend('train', 'val') ;
  set(h,'color','none');
  title('objective') ;
  subplot(1,2,2) ;
  switch opts.errorType
    case 'multiclass'
      plot(1:epoch, info.train.error, 'k') ; hold on ;
      plot(1:epoch, info.train.topFiveError, 'k--') ;
      plot(1:epoch, info.val.error, 'b') ;
      plot(1:epoch, info.val.topFiveError, 'b--') ;
      h=legend('train','train-5','val','val-5') ;
    case 'binary'
      plot(1:epoch, info.train.error, 'k') ; hold on ;
      plot(1:epoch, info.val.error, 'b') ;
      h=legend('train','val') ;
  end
  grid on ;
  xlabel('training epoch') ; ylabel('error') ;
  set(h,'color','none') ;
  title('error') ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res, speed)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
info.speed(end) = info.speed(end) + speed ;
switch opts.errorType
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error(end) = info.error(end) +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError(end) = info.topFiveError(end) + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
end



