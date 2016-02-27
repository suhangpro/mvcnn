function [sample, sampleQueue] = next_samples(sample, sampleQueue, label, nSample, balancingFn)
%NEXT_SAMPLES Shuffling + sampling + balancing + clipping + caching
% 
% Hang Su

if ~exist('sampleQueue', 'var') || isempty(sampleQueue), 
  sampleQueue = [];
end

if ~exist('label', 'var') || isempty(label), 
  label = ones(1,numel(sample));
end

if ~exist('nSample', 'var') || isempty(nSample), 
  nSample = Inf;
end

if ~exist('balancingFn', 'var') || isempty(balancingFn), 
  balancingFn = @(v) v;
end

if isnumeric(balancingFn), 
  balancingFn =  get_default_balancingfn(balancingFn);
end

if numel(sampleQueue)<nSample,
  label_unique = unique(label);
  labelMap = arrayfun(@(v) sample(label==v), ...
    label_unique, 'UniformOutput', false);
  cnt0 = cellfun(@(c) numel(c), labelMap);
  cnt = balancingFn(cnt0);
  sample = [];
  for i=1:numel(cnt),
    if cnt(i)==cnt0(i),
      sample = [sample labelMap{i}];
      continue;
    end
    labelMap{i} = labelMap{i}(randperm(numel(labelMap{i})));
    if cnt(i)<cnt0(i), % sample larger classes
      sample = [sample labelMap{i}(1:cnt(i))];
    else % augment smaller classes
      sample = [sample labelMap{i}(1:mod(cnt(i),cnt0(i)))];
      sample = [sample repmat(labelMap{i},[1 floor(cnt(i)/cnt0(i))])];
    end
  end
  sampleQueue = [sampleQueue sample(randperm(numel(sample)))];
end
sample_end = min(nSample, numel(sampleQueue));
sample = sampleQueue(1:sample_end);
sampleQueue = sampleQueue(sample_end+1:end);

function f = get_default_balancingfn(t)
f = @(v) default_balancingfn(v,t);

function v = default_balancingfn(v,t)
v = round(mean(v)*(v/mean(v)).^t);
