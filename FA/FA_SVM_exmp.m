tic % 计时器
%% 清空环境变量
close all
clear
clc
format compact
%% 数据提取
% 载入测试数据wine,其中包含的数据为classnumber = 3,wine:178*13的矩阵,wine_labes:178*1的列向量
load wine.mat
% 选定训练集和测试集
% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% 相应的训练集的标签也要分离出来
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% 相应的测试集的标签也要分离出来
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
%% 数据预处理
% 数据预处理,将训练集和测试集归一化到[0,1]区间
[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% FA优化参数
% 参数向量 parameters [n N_iteration alpha betamin gamma]
% n为种群规模，N_iteration为迭代次数
para=[10,50,0.5,0.2,1];

% 待优化参数上下界 Simple bounds/limits for d-dimensional problems
d=2; % 待优化参数个数
Lb=[0.01,0.01]; % 下界
Ub=[100,100]; % 上界

% 参数初始化 Initial random guess
u0=Lb+(Ub-Lb).*rand(1,d);

% 迭代寻优 Display results
[bestsolutio,bestojb]=ffa_mincon_svm(@objfun_svm,u0,Lb,Ub,para,train_wine_labels,train_wine,test_wine_labels,test_wine);
%% 打印参数选择结果
bestc=bestsolutio(1);
bestg=bestsolutio(2);

disp('打印选择结果');
str=sprintf('Best c = %g，Best g = %g',bestc,bestg);
disp(str)
%% 利用最佳的参数进行SVM网络训练
cmd_gwosvm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwosvm = svmtrain(train_wine_labels,train_wine,cmd_gwosvm);
%% SVM网络预测
[predict_label,accuracy] = svmpredict(test_wine_labels,test_wine,model_gwosvm);
% 打印测试集分类准确率
total = length(test_wine_labels);
right = sum(predict_label == test_wine_labels);
disp('打印测试集分类准确率');
str = sprintf( 'Accuracy = %g%% (%d/%d)',accuracy(1),right,total);
disp(str);
%% 结果分析
% 测试集的实际分类和预测分类图
figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_label,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on
snapnow
%% 显示程序运行时间
toc