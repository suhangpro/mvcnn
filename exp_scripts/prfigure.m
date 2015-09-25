
recall_i = 0:0.1:1.0;
fh = figure(1);clf;colormap(lines);
pr_wzr = [1.0000    0.5767    0.4589    0.3738    0.3135    0.2703    0.2331    0.2031    0.1722    0.1383    0.0250
1.0000    0.6796    0.5637    0.4768    0.4179    0.3639    0.3143    0.2654    0.2194    0.1662    0.0250
1.0000    0.7184    0.6358    0.5793    0.5322    0.4842    0.4371    0.3928    0.3345    0.2533    0.0250];
plot(recall_i,pr_wzr(1,:),recall_i,pr_wzr(2,:),recall_i,pr_wzr(3,:)); hold on;

eval_mvcnn = load('data/features/l11/evalRet.mat');
eval_mvcnn_metric = load('data/features/l12/evalRet.mat');
eval_fv = load('data/features/l5/evalRet.mat');

info = eval_fv.info;
pr_fv = zeros(size(info.recall_i,1),length(recall_i));
for i=1:size(info.recall_i,1), 
    [recall_all,I] = unique(info.recall_i(i,:));
    precision_all = info.precision_i(i,I);
    pr_fv(i,:) = interp1(recall_all,precision_all,recall_i);
end

info = eval_mvcnn.info;
pr_mvcnn = zeros(size(info.recall_i,1),length(recall_i));
for i=1:size(info.recall_i,1), 
    [recall_all,I] = unique(info.recall_i(i,:));
    precision_all = info.precision_i(i,I);
    pr_mvcnn(i,:) = interp1(recall_all,precision_all,recall_i);
end

info = eval_mvcnn_metric.info;
pr_mvcnn_metric = zeros(size(info.recall_i,1),length(recall_i));
for i=1:size(info.recall_i,1), 
    [recall_all,I] = unique(info.recall_i(i,:));
    precision_all = info.precision_i(i,I);
    pr_mvcnn_metric(i,:) = interp1(recall_all,precision_all,recall_i);
end
plot(recall_i,mean(pr_fv),recall_i,mean(pr_mvcnn),recall_i,mean(pr_mvcnn_metric)); 
grid on; axis square;
legend('Spherical Harmonic','Light Field','3D ShapeNets','Fisher vector', 'Ours (MVCNN)','Ours (MVCNN+metric)');
ylabel('Precision');
xlabel('Recall');

print(fh,'pr.eps','-depsc');
