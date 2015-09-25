function C = plot_confusionmat(gt, pred, labels, savePath, fTitle)
%PLOT_CONFUSIONMAT plot confusion matrix from classification results
%   gt:: 
%       Nx1 ground truth labels
%   pred:: 
%       Nx1 predicted labels
%   labels::
%       Nx1 cell, label names 
%   savePath:: []
%       path to save the figure as pdf file 
%       if specified, figure will be closed automatically
%   fTitle:: 'confusion matrix'
%       title of figure

if ~exist('fTitle','var') || isempty(fTitle), 
    fTitle = 'confusion matrix'; 
end
if ~exist('savePath','var') || isempty(savePath), 
    savePath = [];
end

C = confusionmat(gt,pred);
S = sum(C,2);
S(S==0)=1;
C = bsxfun(@rdivide, C, S);

figure(1); clf;
imagesc(C); colormap(gray);
title(fTitle);
axis square; colorbar;

labels = cellfun(@(s) strrep(s,'_',' '), labels, 'UniformOutput', false); 

% set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
set(gca, 'XTick', 1:length(labels), 'XTickLabel', []);
set(gca, 'YTick', 1:length(labels), 'YTickLabel', labels);

drawnow; 
if ~isempty(savePath), 
    print(1, savePath, '-dpdf'); 
    close(1);
end
