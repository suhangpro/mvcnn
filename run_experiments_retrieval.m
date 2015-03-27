% function run_experiments_retrieval()

setup;
% multiviewOn = {'modelnet10toon', 'modelnet10toonedge', ...
%                 'modelnet40toon', 'modelnet40toonedge'};
trainGpuMode = true;
evalAug = 'none';
skipEval = false; % if true, skip all evaluation
skipTrain = true; % if true, skip all training

models = {};
ex = struct([]);

ex(end+1).model     = 'imagenet-vgg-m';
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalMView   = true;
ex(end).norm       	= false; 
ex(end).evalDataset = { 'modelnet10toon'}; 

ex(end+1).model     = 'imagenet-vgg-verydeep-16';
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalMView   = true;
ex(end).norm        = false;
ex(end).evalDataset = { 'modelnet10toon'}; 

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'modelnet10toon';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).trainMView  = true;
ex(end).numEpochs   = 15;
ex(end).featLayer   = 'fc7';
ex(end).evalGpuMode = false;
ex(end).evalMView   = true;
ex(end).norm        = false; 
ex(end).evalDataset = { 'modelnet10toon'}; 

for i=1:length(ex), 
    % train / fine-tune 
    if ~isfield(ex(i),'model') || isempty(ex(i).model), 
        prefix = sprintf('BS%d_AUG%s_MV%d', ...
            ex(i).batchSize, ex(i).trainAug, ex(i).trainMView);
        ex(i).model = sprintf('%s-finetuned-%s-%s', ex(i).baseModel, ...
            ex(i).trainDataset, prefix);
        if ~exist(fullfile('data','models',[ex(i).model '.mat']),'file'),
            if skipTrain, continue; end; 
            net = run_train(ex(i).trainDataset, ...
                'modelName', ex(i).baseModel,...
                'numEpochs', ex(i).numEpochs, ...
                'prefix', prefix, ...
                'batchSize', ex(i).batchSize, ...
                'augmentation', ex(i).trainAug, ...
                'multiview', ex(i).trainMView, ...
                'gpuMode', trainGpuMode);
            models{end+1} = ex(i).model;
            save(fullfile('data','models',[model '.mat']),'-struct','net');
        end
    end
    % compute and evaluate features 
    if isfield(ex(i),'evalDataset') && ~isempty(ex(i).evalDataset) && ~skipEval, 
        for dataset = ex(i).evalDataset, 
            if ex(i).norm, 
                suffix = sprintf('NORM%d-PCA%d',ex(i).norm,ex(i).pca);
            else
                suffix = sprintf('NORM%d',ex(i).norm);
                ex(i).pca = Inf;
            end
            featDir = fullfile('data', 'features', ...
                [dataset{1} '-' ex(i).model '-' evalAug], suffix);
            % skip the evaluation if feature ready exists
            if exist(fullfile(featDir, [ex(i).featLayer '.mat']),'file'), 
                continue;
            end
            if ~ex(i).evalGpuMode, poolObj = gcp(); end;
            feats = imdb_compute_cnn_features(dataset{1}, ex(i).model, ...
                'augmentation', evalAug, ...
                'gpuMode', ex(i).evalGpuMode, ...
                'normalization', ex(i).norm, ...
                'pca', ex(i).pca);
            feat = feats.(ex(i).featLayer);
        end
    end
end

