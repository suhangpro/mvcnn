addpath('liblinear-1.96/matlab/');

imdb = load('data/v1/sketch-seed-01/imdb/imdb-seed-1.mat');
load('data/v1/sketch-seed-01/fc7.mat');

trainIdxs   = find(imdb.images.set==1 | imdb.images.set==2);
testIdxs    = find(imdb.images.set==3);

trainLabel  = imdb.images.label(trainIdxs)';
testLabel   = imdb.images.label(testIdxs)';

trainFeat   = sparse(feat(:,trainIdxs));
testFeat    = sparse(feat(:,testIdxs));

bestcv = 0;
for log2c = -4:2:4,
	cmd = ['-v 5 -q -c ', num2str(2^log2c)];
    cv = liblinear_train(trainLabel,trainFeat,cmd,'col');
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
end

cmd = ['-q -c ', num2str(bestc)];
model = liblinear_train(trainLabel,trainFeat,cmd,'col');

cmd = ['-q'];
[~,accuracy,~] = liblinear_predict(testLabel,testFeat,model,cmd,'col');
fprintf('Accuracy on sketch dataset (val): %g\n',bestcv);
fprintf('Accuracy on sketch dataset (test): %g\n',accuracy(1));
