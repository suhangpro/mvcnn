function run_evaluate(feat, varargin)
%RUN_EVALUATE Evaluate CNN activations features
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
%   `multiview`:: false
%       set to true to evaluate on multiple views of same instances 
%   `logPath`:: 'log/eval.txt'
%       place to save log information
%   `predPath`:: 'data/pred.mat'
%       place to save prediction results
%   `confusionPath`:: 'data/confusion.pdf' 
%       place to save confusion matrix plot 
%   
opts.imdb = [];
opts.cv = 5;
opts.log2c = [-4:2:4];
opts.quiet = true;
opts.multiview = false;
opts.logPath = fullfile('log','eval.txt');
opts.predPath = fullfile('data','pred.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.confusionPath = fullfile(fileparts(opts.predPath), 'confusion.pdf');
opts = vl_argparse(opts, varargin) ;

if ~exist(fileparts(opts.logPath),'dir'), 
    vl_xmkdir(fileparts(opts.logPath)); 
end
if ~exist(fileparts(opts.predPath),'dir'), 
    vl_xmkdir(fileparts(opts.predPath)); 
end

if ischar(feat), 
    feat = load(feat);
end

if isfield(feat,'imdb'), 
    imdb = feat.imdb; 
else
    imdb = opts.imdb;
end

trainIdxs   = find(imdb.images.set==1 | imdb.images.set==2)';
testIdxs    = find(imdb.images.set==3)';

trainLabel  = imdb.images.class(trainIdxs)';
testLabel   = imdb.images.class(testIdxs)';

% only keep training samples from same classes with testing instances 
labelIdxs   = sort(unique(testLabel));
nClasses    = length(labelIdxs); 
trainIdxs   = trainIdxs(ismember(trainLabel,labelIdxs));
trainLabel  = imdb.images.class(trainIdxs)';

trainFeat   = sparse(feat.x(trainIdxs,:));
testFeat    = sparse(feat.x(testIdxs,:));


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
    
    save(opts.predPath,'bestc','accuTrain','predTest','accuTest','decTest',...
        'model');
end

if opts.multiview, 
    imNames = cellfun(@(s) parse_name_with_view(s), ...
        imdb.images.name(testIdxs)', 'UniformOutput', false); 
    imNamesUnique = unique(imNames); 
    nUniques = length(imNamesUnique); 
    [~, imIds] = ismember(imNames, imNamesUnique); 
    
    testLabel0 = testLabel;
    testLabel = zeros(nUniques,1);
    for i=1:nUniques, 
        testLabel(i) = testLabel0(find(imIds==i,1));
    end
    
    %{
    histCnt = zeros(nUniques, nClasses);
    for ci = 1:nClasses, 
        c = labelIdxs(ci); 
        histCnt(:,ci) = histc(imIds(predTest==c),1:nUniques);
    end
    decTest = bsxfun(@rdivide, histCnt, sum(histCnt,2)); % normalize
    %}
    
    %{:-)
    decSum  = zeros(nUniques, nClasses); 
    cnt = zeros(nUniques,1);
    for i = 1:nUniques, 
        decSum(i,:) = sum(decTest(imIds==i,:));
        cnt(i) = sum(imIds==i); 
    end
    decTest = bsxfun(@rdivide, decSum, cnt); % average 
    %}
    
    [~,I] = max(decTest,[],2);
    predTest = labelIdxs(I);
    accuTest = sum(predTest==testLabel)/length(predTest); 
    
end

% compute mAP for test set
AP = zeros(1,nClasses);
for ci=1:nClasses,
    [~,~,info] = vl_pr((testLabel==labelIdxs(ci))-0.5,decTest(:,ci));
    AP(ci) = info.ap;
end
mAP = mean(AP);

% confusion matrix
plot_confusionmat(testLabel, predTest, imdb.meta.classes(labelIdxs), ...
    opts.confusionPath, [imdb.imageDir ' : ' feat.modelName]);

fprintf('Evaluation finished! \n');
fprintf('\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf('\tdataset: %s\n', imdb.imageDir);
fprintf('\tmodel: %s\n',feat.modelName);
fprintf('\tlayer: %s\n',feat.layerName);
fprintf('\taccuracy (train): %g%%\n',accuTrain*100);
fprintf('\taccuracy (test): %g%%\n',accuTest*100);
fprintf('\tmAP (test): %g\n\n',mAP);

fid = fopen(opts.logPath,'a+');
fprintf(fid, '(%s) \n', datestr(now));
fprintf(fid, '\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf(fid, '\tdataset: %s\n', imdb.imageDir);
fprintf(fid, '\tmodel: %s\n',feat.modelName);
fprintf(fid, '\tlayer: %s\n',feat.layerName);
fprintf(fid, '\taccuracy (train): %g%%\n',accuTrain*100);
fprintf(fid, '\taccuracy (test): %g%%\n',accuTest*100);
fprintf(fid, '\tmAP (test): %g\n\n',mAP);
fclose(fid);

function [name, view] = parse_name_with_view(filename)
[pathstr, name0, ~] = fileparts(filename); 
name0 = fullfile(pathstr, name0); 
idx = strfind(name0,'_'); 
view = [];
if isempty(idx) || isempty(str2double(name0(idx(end)+1:end))), 
    name = name0;
    return;
end
name = name0(1:idx(end)-1);
view = str2num(name0(idx(end)+1:end));
