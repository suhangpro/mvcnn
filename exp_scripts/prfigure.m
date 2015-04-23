recall_i = 0:0.1:1.0;
figure(1);
clf;
pr10 = [1.0000    0.7188    0.6138    0.5448    0.4585    0.4088    0.3628    0.3134    0.2490    0.1801    0.1000
1.0000    0.7686    0.6768    0.6115    0.5550    0.4936    0.4334    0.3797    0.2919    0.2054    0.1000
1.0000    0.8436    0.8054    0.7823    0.7591    0.7330    0.7006    0.6510    0.5768    0.4183    0.1000];

pr40 = [1.0000    0.5767    0.4589    0.3738    0.3135    0.2703    0.2331    0.2031    0.1722    0.1383    0.0250
    1.0000    0.6796    0.5637    0.4768    0.4179    0.3639    0.3143    0.2654    0.2194    0.1662    0.0250
    1.0000    0.7184    0.6358    0.5793    0.5322    0.4842    0.4371    0.3928    0.3345    0.2533    0.0250];

subplot(1,2,1); 
plot(recall_i,pr10(1,:),recall_i,pr10(2,:),recall_i,pr10(3,:));
load('~/Desktop/viewpool-retrieval.mat');
hold on;plot(mean(info.recall_i),mean(info.precision_i));
ylabel('Precision');
xlabel('Recall');
title('10 Classes Results','FontSize',20);
legend({'Spherical Harmonic','Light Field','3D ShapeNets','Ours'},'FontSize',20,'Location','southwest');
axis square; grid on;

subplot(1,2,2); 
plot(recall_i,pr40(1,:),recall_i,pr40(2,:),recall_i,pr40(3,:));
load('~/Desktop/viewpool-retrieval-40.mat');
hold on;plot(mean(info.recall_i),mean(info.precision_i));
ylabel('Precision');
xlabel('Recall');
title('40 Classes Results');
legend('Spherical Harmonic','Light Field','3D ShapeNets','Ours');
axis square; grid on;
