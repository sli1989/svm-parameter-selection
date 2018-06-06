%% Objective Function
function f=objfun_svm(cv,train_wine_labels,train_wine,test_wine_labels,test_wine)
% cv为长度为2的横向量，即SVM中参数c和v的值

cmd = [' -c ',num2str(cv(1)),' -g ',num2str(cv(2))];
model=svmtrain(train_wine_labels,train_wine,cmd); % SVM模型训练
[~,fitness]=svmpredict(test_wine_labels,test_wine,model); % SVM模型预测及其精度
f=1-fitness(1)/100; % 以分类预测错误率作为优化的目标函数值