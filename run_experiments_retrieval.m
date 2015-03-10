% function run_experiments_retrieval()

setup;
multiviewOn = {'modelnet40toon', 'modelnet40toonedge'};
trainGpuMode = true;
evalAug = 'none';
evalOnly = true; % if true, skip all training

models = {};
ex = struct([]);

ex(end+1).model     = 'imagenet-vgg-m';
ex(end).featLayer   = 'fc6'; 
ex(end).evalGpuMode = false;
ex(end).pca         = 500; 
ex(end).evalDataset = { 'modelnet10toon'}; 

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'modelnet10toon';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).addDropout  = false; % TODO: try different options 
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc6';
ex(end).evalGpuMode = false;
ex(end).pca         = 500; 
ex(end).evalDataset = { 'modelnet10toon'}; 

ex(end+1).model     = 'imagenet-vgg-m';
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).pca         = 500; 
ex(end).evalDataset = { 'modelnet10toon'}; 

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'modelnet10toon';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).addDropout  = false; 
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7';
ex(end).evalGpuMode = false;
ex(end).pca         = 500; 
ex(end).evalDataset = { 'modelnet10toon'}; 

for i=1:length(ex), 
    % train / fine-tune 
    if ~isfield(ex(i),'model') || isempty(ex(i).model), 
        prefix = sprintf('BS%d_AUG%s', ex(i).batchSize, ex(i).trainAug);
        ex(i).model = sprintf('%s-finetuned-%s-%s', ex(i).baseModel, ...
            ex(i).trainDataset, prefix);
        if ~exist(fullfile('data','models',[ex(i).model '.mat']),'file'),
            if evalOnly, continue; end; 
            net = run_train(ex(i).trainDataset, ...
                'modelName', ex(i).baseModel,...
                'numEpochs', ex(i).numEpochs, ...
                'prefix', prefix, ...
                'batchSize', ex(i).batchSize, ...
                'augmentation', ex(i).trainAug, ...
                'addDropout', ex(i).addDropout, ...
                'gpuMode', trainGpuMode);
            models{end+1} = ex(i).model;
            save(fullfile('data','models',[model '.mat']),'-struct','net');
        end
    end
    % compute and evaluate features 
    if isfield(ex(i),'evalDataset') && ~isempty(ex(i).evalDataset), 
        for dataset = ex(i).evalDataset, 
            suffix = sprintf('NORM%d-PCA%d',true,ex(i).pca);
            featDir = fullfile('data', 'features', ...
                [dataset{1} '-' ex(i).model '-' evalAug], suffix);
            if exist(fullfile(featDir, [ex(i).featLayer '.mat']),'file'), 
                feat = load(fullfile(featDir, [ex(i).featLayer '.mat']));
            else
                if ~ex(i).evalGpuMode, poolObj = gcp(); end;
                feats = imdb_compute_cnn_features(dataset{1}, ex(i).model, ...
                    'augmentation', evalAug, ...
                    'gpuMode', ex(i).evalGpuMode, ...
                    'normalization', true, ...
                    'pca', ex(i).pca);
                feat = feats.(ex(i).featLayer);
            end
        end
    end
end

