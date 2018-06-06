% 清除变量空间
clc
clear all
close all
% 生成训练数据
train_data = linspace(-2.2,2.2,50)';
train_label = sinc(train_data) +normrnd(0,.1,size(train_data,1),1);

% ABC算法优化SVM参数
% [BestMSE,BestParams,ABCOpts] = ABCSVMcgpForRegress(train_label,train_data);
[BestMSE,BestParams,ABCOpts] = ABCSVMcgForRegress(train_label,train_data);

% 训练并对训练集回归预测
% cmd = [' -c ',num2str(BestParams(1)),' -g ',num2str(BestParams(2)),' -p ',num2str(BestParams(3)),' -s 3'];
cmd = [' -c ',num2str(BestParams(1)),' -g ',num2str(BestParams(2)),' -s 3 -p 0.01'];

model = svmtrain(train_label, train_data,cmd);
[ptrain, train_mse] = svmpredict(train_label, train_data, model);

% 预测结果可视化
figure;
plot(train_label,'-o');
hold on;
plot(ptrain,'r-s');
grid on;
legend('original','predict');
title('Train Set Regression Predict by SVM');
hold off
