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
%   `logDir`:: 'log'
%       place to save log file (eval.txt) 
%   
opts.imdb = [];
opts.cv = 5;
opts.log2c = [-4:2:4];
opts.quiet = true;
opts.logDir = 'log';
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.logDir,'dir'), 
    vl_xmkdir(opts.logDir); 
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
[~,accuTest,~] = liblinear_predict(testLabel,testFeat,model,cmd);

fprintf('Evaluation finished! \n');
fprintf('\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf('\tdataset: %s\n', imdb.imageDir);
fprintf('\tmodel: %s\n',feat.modelName);
fprintf('\tlayer: %s\n',feat.layerName);
fprintf('\taccuracy (train): %g\n',accuTrain(1));
fprintf('\taccuracy (test): %g\n\n',accuTest(1));

fid = fopen(fullfile(opts.logDir,'eval.txt'),'a+');
fprintf(fid, '(%s) \n', datestr(now));
fprintf(fid, '\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf(fid, '\tdataset: %s\n', imdb.imageDir);
fprintf(fid, '\tmodel: %s\n',feat.modelName);
fprintf(fid, '\tlayer: %s\n',feat.layerName);
fprintf(fid, '\taccuracy (train): %g\n',accuTrain(1));
fprintf(fid, '\taccuracy (test): %g\n\n',accuTest(1));
