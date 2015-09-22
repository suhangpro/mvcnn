function evaluate_classification(feat, varargin)
%EVALUATE_CLASSIFICATION Evaluate CNN features for classification 
%
%   feat::
%       a structure containing cnn feature
%   `imdb`:: []
%       optional, usually a field in feat
%   `cv`:: 5
%       #folds in cross validation (-v)
%   `log2c`:: [-4:2:4]
%       tunable liblinear svm parameter (-c) 
%   `quiet`:: true
%       liblinear parameter (-q)
%   `multiview`:: true
%       set to false to evaluate on every single views of same instances 
%   `method`:: 'avgdesc'
%       used only if multiview is true; other choices: 'maxdesc','avgsvmscore'
%   `logPath`:: 'log/eval.txt'
%       place to save log information
%   `predPath`:: 'data/pred.mat'
%       place to save prediction results
%   `confusionPath`:: 'data/confusion.pdf' 
%       place to save confusion matrix plot 
%   
% NOTE: assume all classes in imdb appear in training set 

% default options 
opts.imdb = [];
opts.cv = 5;
opts.log2c = [-4:2:4];
opts.quiet = true;
opts.multiview = true;
opts.method = 'avgsdesc';
opts.logPath = fullfile('log','eval.txt');
opts.predPath = fullfile('data','pred.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.confusionPath = fullfile(fileparts(opts.predPath), 'confusion.pdf');
[opts, varargin] = vl_argparse(opts, varargin) ;

if ~exist(fileparts(opts.logPath),'dir'), 
    vl_xmkdir(fileparts(opts.logPath)); 
end
if ~exist(fileparts(opts.predPath),'dir'), 
    vl_xmkdir(fileparts(opts.predPath)); 
end
if ischar(feat), 
    feat = load(feat);
end

% -------------------------------------------------------------------------
%                       sort imdb.images & feat.x w.r.t (sid,view) or (id)
% -------------------------------------------------------------------------
if isfield(feat,'imdb'), 
    imdb = feat.imdb; 
else
    imdb = opts.imdb;
end
if ~isfield(imdb.images,'sid'), 
    nInstances = numel(imdb.images.name); 
else
    nInstances = length(unique(imdb.images.sid));
end

% sort imdb.images wrt id
[imdb.images.id,I] = sort(imdb.images.id);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);
if isfield(imdb.images,'sid'), imdb.images.sid = imdb.images.sid(I); end

% sort feat.x wrt id/sid
if isfield(feat, 'sid'), 
    pooledFeat = true;
    opts.multiview = false;
    [feat.sid,I] = sort(feat.sid);
    feat.x = feat.x(I,:);
else
    pooledFeat = false;
    [feat.id,I] = sort(feat.id);
    feat.x = feat.x(I,:);
end

% sort imdb.images wrt sid
if isfield(imdb.images,'sid'),
    [imdb.images.sid, I] = sort(imdb.images.sid);
    imdb.images.name = imdb.images.name(I);
    imdb.images.class = imdb.images.class(I);
    imdb.images.set = imdb.images.set(I);
    imdb.images.id = imdb.images.id(I);
    if ~pooledFeat, feat.x = feat.x(I,:); end
end

% -------------------------------------------------------------------------
%                                                      feature descriptors
% -------------------------------------------------------------------------
nViews = length(imdb.images.name)/nInstances;
nDescPerShape = size(feat.x,1)/nInstances;
shapeGtClasses = imdb.images.class(1:nViews:end);
shapeSets = imdb.images.set(1:nViews:end);
nDims = size(feat.x,2);

% train & val
trainSets = {'train','val'};
testSets = {'test'};
[~,I] = ismember(trainSets,imdb.meta.sets);
trainSids = find(ismember(shapeSets, I));
tmp = zeros(nDescPerShape, nInstances);
tmp(:,trainSids) = 1;
trainFeat = feat.x(find(tmp)',:);
trainLabel = shapeGtClasses(trainSids)';
nTrainShapes = length(trainLabel); 

% test 
[~,I] = ismember(testSets,imdb.meta.sets);
testSids = find(ismember(shapeSets, I));
tmp = zeros(nDescPerShape, nInstances);
tmp(:,testSids) = 1;
testFeat = feat.x(find(tmp)',:);
testLabel = shapeGtClasses(testSids)';
nTestShapes = length(testLabel);

if ~pooledFeat, 
    if opts.multiview && strcmp(opts.method,'avgdesc'), 
        % average descriptor across views 
        trainFeat = reshape(mean(reshape(trainFeat, ...
            [nDescPerShape nTrainShapes*nDims]),1),[nTrainShapes nDims]);
        testFeat = reshape(mean(reshape(testFeat, ...
            [nDescPerShape nTestShapes*nDims]),1),[nTestShapes nDims]);
    elseif opts.multiview && strcmp(opts.method,'maxdesc'), 
        % max descriptor across views 
        trainFeat = reshape(max(reshape(trainFeat, ...
            [nDescPerShape nTrainShapes*nDims]),[],1),[nTrainShapes nDims]);
        testFeat = reshape(max(reshape(testFeat, ...
            [nDescPerShape nTestShapes*nDims]),[],1),[nTestShapes nDims]);
    else
        % expand labels across views 
        trainLabel = reshape(repmat(trainLabel',[nDescPerShape 1]),...
            [nDescPerShape*nTrainShapes 1]);
        testLabel = reshape(repmat(testLabel',[nDescPerShape 1]),...
            [nDescPerShape*nTestShapes 1]);
    end
end   
trainFeat = sparse(double(trainFeat));
testFeat = sparse(double(testFeat)); 


% -------------------------------------------------------------------------
%                                                   train and evaluate SVM
% -------------------------------------------------------------------------
if exist(opts.predPath, 'file'), 
    load(opts.predPath); 
    fprintf('SVM model and predictions loaded from %s. \n', opts.predPath);
else
    fprintf('Evaluating ... \n');
    
    bestcv = 0;
    for log2c = opts.log2c,
        cmd = ['-v ', num2str(opts.cv) ,' -c ', num2str(2^log2c)];
        if opts.quiet, cmd = [cmd ' -q']; end;
        cv = liblinear_train(trainLabel,trainFeat,cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c;
        end
        fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cmd = ['-c ', num2str(bestc)];
    if opts.quiet, cmd = [cmd ' -q']; end;
    model = liblinear_train(trainLabel,trainFeat,cmd);
    
    cmd = [''];
    if opts.quiet, cmd = [cmd ' -q']; end;
    [~,accuTrain,~] = liblinear_predict(trainLabel,trainFeat,model,cmd);
    [predTest,accuTest,decTest] = liblinear_predict(testLabel,testFeat,...
        model,cmd);
    [~,I] = sort(model.Label);
    decTest = decTest(:,I);
    accuTest = accuTest(1)/100;
    accuTrain = accuTrain(1)/100;
    
    if opts.multiview && strcmp(opts.method,'avgsvmscore'),
        testLabel = testLabel(1:nDescPerShape:end);
        nClasses = length(model.Label);
        decTest = reshape(mean(reshape(decTest, ...
            [nDescPerShape nTestShapes*nClasses]),1),[nTestShapes nClasses]);
        [~,predTest] = max(decTest,[],2);
        accuTest = sum(predTest==testLabel)/length(predTest);
    end

    save(opts.predPath,'bestc','accuTrain','predTest','accuTest','decTest',...
        'model','opts');
end

% -------------------------------------------------------------------------
%                                                   write & output results 
% -------------------------------------------------------------------------

% confusion matrix
plot_confusionmat(testLabel, predTest, imdb.meta.classes, ...
    opts.confusionPath, [imdb.imageDir ' : ' feat.modelName]);

STR_01 = {'false', 'true'};
fprintf('Evaluation finished! \n');
fprintf('\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf('\tdataset: %s\n', imdb.imageDir);
fprintf('\tmodel: %s\n',feat.modelName);
fprintf('\tlayer: %s\n',feat.layerName);
fprintf('\tmultiview: %s', STR_01{(opts.multiview~=0)+1});
if opts.multiview, fprintf(' (%s)', opts.method); end; fprintf('\n');
fprintf('\taccuracy (train): %g%%\n',accuTrain*100);
fprintf('\taccuracy (test): %g%%\n',accuTest*100);

fid = fopen(opts.logPath,'a+');
fprintf(fid, '(%s) -- Classification\n', datestr(now));
fprintf(fid, '\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf(fid, '\tdataset: %s\n', imdb.imageDir);
fprintf(fid, '\tmodel: %s\n',feat.modelName);
fprintf(fid, '\tlayer: %s\n',feat.layerName);
fprintf(fid, '\tmultiview: %s', STR_01{(opts.multiview~=0)+1});
if opts.multiview, fprintf(fid,' (%s)', opts.method); end; fprintf(fid,'\n');
fprintf(fid, '\taccuracy (train): %g%%\n',accuTrain*100);
fprintf(fid, '\taccuracy (test): %g%%\n',accuTest*100);
fclose(fid);
