%% Objective Function
function f=objfun_svm(cv,train_wine_labels,train_wine,test_wine_labels,test_wine)
% cvΪ����Ϊ2�ĺ���������SVM�в���c��v��ֵ

cmd = [' -c ',num2str(cv(1)),' -g ',num2str(cv(2))];
model=svmtrain(train_wine_labels,train_wine,cmd); % SVMģ��ѵ��
[~,fitness]=svmpredict(test_wine_labels,test_wine,model); % SVMģ��Ԥ�⼰�侫��
f=1-fitness(1)/100; % �Է���Ԥ���������Ϊ�Ż���Ŀ�꺯��ֵ