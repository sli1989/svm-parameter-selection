% ��������ռ�
clc
clear all
close all
% ����ѵ������
train_data = linspace(-2.2,2.2,50)';
train_label = sinc(train_data) +normrnd(0,.1,size(train_data,1),1);

% ABC�㷨�Ż�SVM����
% [BestMSE,BestParams,ABCOpts] = ABCSVMcgpForRegress(train_label,train_data);
[BestMSE,BestParams,ABCOpts] = ABCSVMcgForRegress(train_label,train_data);

% ѵ������ѵ�����ع�Ԥ��
% cmd = [' -c ',num2str(BestParams(1)),' -g ',num2str(BestParams(2)),' -p ',num2str(BestParams(3)),' -s 3'];
cmd = [' -c ',num2str(BestParams(1)),' -g ',num2str(BestParams(2)),' -s 3 -p 0.01'];

model = svmtrain(train_label, train_data,cmd);
[ptrain, train_mse] = svmpredict(train_label, train_data, model);

% Ԥ�������ӻ�
figure;
plot(train_label,'-o');
hold on;
plot(ptrain,'r-s');
grid on;
legend('original','predict');
title('Train Set Regression Predict by SVM');
hold off
