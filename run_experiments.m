% function run_experiments()

setup;
multiviewOn = {'modelnet40toon', 'modelnet40toonedge'};
trainGpuMode = true;
logPath = fullfile('log','eval1.txt'); 
evalOnly = true; % if true, skip all training

models = {};
ex = struct([]);

ex(end+1).model     = 'imagenet-vgg-m';
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'clipart100', ...
                        'sketch160', ...
                        'modelnet40toon', ...
                        'modelnet40toonedge'};

ex(end+1).model     = 'imagenet-vgg-verydeep-16'; 
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'clipart100', ...
                        'sketch160', ...
                        'modelnet40toon', ...
                        'modelnet40toonedge'};

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'clipart100gpb';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'sketch160'};

ex(end+1).baseModel = 'imagenet-vgg-verydeep-16';
ex(end).trainDataset= 'clipart100gpb';
ex(end).batchSize   = 32;
ex(end).trainAug    = 'none';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'sketch160'};

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'sketch160';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'sketch160'};

ex(end+1).baseModel = 'imagenet-vgg-verydeep-16';
ex(end).trainDataset= 'sketch160';
ex(end).batchSize   = 32;
ex(end).trainAug    = 'none';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'sketch160'};

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'modelnet40toonedge';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'sketch160', ...
                        'modelnet40toonedge'};

ex(end+1).baseModel = 'imagenet-vgg-verydeep-16';
ex(end).trainDataset= 'modelnet40toonedge';
ex(end).batchSize   = 32;
ex(end).trainAug    = 'none';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100gpb', ...
                        'sketch160', ...
                        'modelnet40toonedge'};

ex(end+1).baseModel = 'imagenet-vgg-m';
ex(end).trainDataset= 'modelnet40toon';
ex(end).batchSize   = 64;
ex(end).trainAug    = 'f2';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100', ...
                        'modelnet40toon'};

ex(end+1).baseModel = 'imagenet-vgg-verydeep-16';
ex(end).trainDataset= 'modelnet40toon';
ex(end).batchSize   = 32;
ex(end).trainAug    = 'none';
ex(end).numEpochs   = 20;
ex(end).featLayer   = 'fc7'; 
ex(end).evalGpuMode = false;
ex(end).evalDataset = {'clipart100', ...
                        'modelnet40toon'};

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
                'gpuMode', trainGpuMode);
            models{end+1} = ex(i).model;
            save(fullfile('data','models',[model '.mat']),'-struct','net');
        end
    end
    % compute and evaluate features 
    if isfield(ex(i),'evalDataset') && ~isempty(ex(i).evalDataset), 
        for dataset = ex(i).evalDataset, 
            featDir = fullfile('data', 'features', [dataset{1} '-' ex(i).model '-none']);
            % skip the evaluation if feature ready exists
            if exist(fullfile(featDir, [ex(i).featLayer '.mat']),'file'), 
                continue; 
            end
            if ~ex(i).evalGpuMode, poolObj = gcp(); end;
            featCell = imdb_compute_cnn_features(dataset{1}, ex(i).model, ...
                'augmentation', 'none', ...
                'gpuMode', ex(i).evalGpuMode, ...
                'normalization', false, ...
                'layers', {ex(i).featLayer});
            run_evaluate(featCell{1}, ...
                'cv', 2, ...
                'logPath', logPath, ...
                'predPath', fullfile(featDir,'pred.mat'), ...
                'log2c', [-8:4:4], ...
                'multiview', ismember(dataset{1},multiviewOn));
        end
    end
end

