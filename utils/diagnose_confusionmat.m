function diagnose_confusionmat(gt, pred, labels)
%DIAGNOSE_CONFUSIONMAT diagnose from classification results
%   gt:: 
%       Nx1 ground truth labels
%   pred:: 
%       Nx1 predicted labels
%   labels::
%       Nx1 cell, label names 

labels = cellfun(@(s) strrep(s,'_',' '), labels, 'UniformOutput', false); 

C = confusionmat(gt,pred);
S = sum(C,2);
C0 = C.*(1-eye(size(C)));
S0 = sum(C0,2);

fprintf('\n   DIAGNOSIS   \n===============\n\n');

fprintf('There are %d testing instances coming from %d categories. \n', ...
    sum(S), length(S));

fprintf('There are %d wrongly classified instances, the overall \naccuracy is %.2f%%. \n\n',...
    sum(S0), (1-sum(S0)/sum(S))*100);

fprintf('   class   ::#test  #error     most-confused-to \n');
fprintf('--------------------------------------------------\n');
for c = 1: length(S), 
    [y,i] = sort(C0(c,:),'descend');
    fprintf('%10s :: %3d %2d(%4.1f%%)', ...
        labels{c}(1:min(10,length(labels{c}))), S(c), S0(c), 100*S0(c)/S(c));
    if y(1)>0, 
        fprintf(' %10s:%3d(%5.1f%%)', ...
            labels{i(1)}(1:min(10,length(labels{i(1)}))), y(1), 100*y(1)/S0(c));
    end
    
    fprintf('\n');
end
fprintf('\n');

C2 = triu(C0+C0');
[Y,I] = sort(C2(:),'descend');
[I1, I2] = ind2sub(size(C),I);
nPairs = min(20,max(find(Y,1,'last')));

fprintf('Most confused pairs: \n\n');
fprintf('            pair              #error(%%)  accum \n');
fprintf('--------------------------------------------------\n');
for p = 1:nPairs, 
    
    fprintf('%12s <=>%12s %3d(%4.1f%%) %5.1f%%\n', ...
        labels{I1(p)}(1:min(12,length(labels{I1(p)}))), ...
        labels{I2(p)}(1:min(12,length(labels{I2(p)}))), ...
        Y(p), 100*Y(p)/sum(S0), 100*sum(Y(1:p))/sum(S0));
end
fprintf('\n');