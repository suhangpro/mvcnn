function run_evaluate(feat, varargin)
%RUN_EVALUATE Evaluate CNN activations features
%
%   feat::
%       a structure containing cnn feature
%   `imdb`:: []
%       optional, usually a field in feat
%   `cv`:: 5
%       #folds in cross validation
%   `log2c`:: [-4:2:4]
%       tunable liblinear svm parameter (-c) 
%   `logDir`:: 'log'
%       place to save log file (eval.txt) 
%   
opts.imdb = [];
opts.cv = 5;
opts.log2c = [-4:2:4];
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

bestcv = 0;
for log2c = opts.log2c,
	cmd = ['-v ', num2str(opts.cv) ,' -q -c ', num2str(2^log2c)];
    cv = liblinear_train(trainLabel,trainFeat,cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
end

cmd = ['-q -c ', num2str(bestc)];
model = liblinear_train(trainLabel,trainFeat,cmd);

cmd = ['-q'];
[~,accuracy,~] = liblinear_predict(testLabel,testFeat,model,cmd);

fprintf('(%s) Evaluation finished.\n', datestr(now));
fprintf('\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf('\tdataset: %s\n', imdb.imageDir);
fprintf('\tmodel: %s\n',feat.modelName);
fprintf('\tlayer: %s\n',feat.layerName);
fprintf('\taccuracy (val): %g\n',bestcv);
fprintf('\taccuracy (test): %g\n\n',accuracy(1));

fid = fopen(fullfile(opts.logDir,'eval.txt'),'a+');
fprintf(fid, '(%s) Evaluation finished.\n', datestr(now));
fprintf(fid, '\tc: %g (cv=%d)\n', bestc, opts.cv);
fprintf(fid, '\tdataset: %s\n', imdb.imageDir);
fprintf(fid, '\tmodel: %s\n',feat.modelName);
fprintf(fid, '\tlayer: %s\n',feat.layerName);
fprintf(fid, '\taccuracy (val): %g\n',bestcv);
fprintf(fid, '\taccuracy (test): %g\n\n',accuracy(1));
