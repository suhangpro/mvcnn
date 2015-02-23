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
%   `logPath`:: 'log/eval.txt'
%       place to save log information
%   `predPath`:: 'data/pred.mat'
%       place to save prediction results
%   
opts.imdb = [];
opts.cv = 5;
opts.log2c = [-4:2:4];
opts.quiet = true;
opts.logPath = fullfile('log','eval.txt');
opts.predPath = fullfile('data','pred.mat');
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


trainIdxs   = find(imdb.images.set==1 | imdb.images.set==2);
testIdxs    = find(imdb.images.set==3);

trainLabel  = imdb.images.class(trainIdxs)';
testLabel   = imdb.images.class(testIdxs)';

trainFeat   = sparse(feat.x(trainIdxs,:));
testFeat    = sparse(feat.x(testIdxs,:));

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
[predTest,accuTest,decTest] = liblinear_predict(testLabel,testFeat,model,cmd);
[~,I] = sort(model.Label);
decTest = decTest(:,I);
save(opts.predPath,'model','predTest','accuTest','decTest');

% compute mAP for testset
AP = zeros(1,length(I));
for c=1:length(AP),
    [~,~,info] = vl_pr((testLabel==c)-0.5,decTest(:,c));
    AP(c) = info.ap;
end
mAP = mean(AP);

fprintf('Evaluation finished! \n');
fprintf('\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf('\tdataset: %s\n', imdb.imageDir);
fprintf('\tmodel: %s\n',feat.modelName);
fprintf('\tlayer: %s\n',feat.layerName);
fprintf('\taccuracy (train): %g\n',accuTrain(1));
fprintf('\taccuracy (test): %g\n\n',accuTest(1));
fprintf('\tmAP (test): %g\n\n',mAP);

fid = fopen(opts.logPath,'a+');
fprintf(fid, '(%s) \n', datestr(now));
fprintf(fid, '\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf(fid, '\tdataset: %s\n', imdb.imageDir);
fprintf(fid, '\tmodel: %s\n',feat.modelName);
fprintf(fid, '\tlayer: %s\n',feat.layerName);
fprintf(fid, '\taccuracy (train): %g\n',accuTrain(1));
fprintf(fid, '\tmAP (test): %g\n\n',mAP);
fclose(fid);
