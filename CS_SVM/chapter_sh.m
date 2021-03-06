%% Matlab神经网络43个案例分析

% 基于SVM的回归预测分析——上证指数开盘指数预测
% by 李洋(faruto)
% http://www.matlabsky.com
% Email:faruto@163.com
% http://weibo.com/faruto 
% http://blog.sina.com.cn/faruto
% 2013.01.01
%% 清空环境变量
function chapter_sh  %函数只能调用，不能运行，脚本M文件可运行，函数只有输出变量在工作空间显示，中间变量作为局部变量，只在函数调用时起作用，不会在工作空间显示
tic;
close all;%close all 是关闭所有窗口(程序运行产生的，不包括命令窗，editor窗和帮助窗）% clear all是清除所有工作空间中的变
clear;
clc;
format compact;  %数据显示的一种样式;空格比较少,数字排列比较紧凑;
%% 数据的提取和预处理                            

% 载入测试数据上证指数(1990.12.19-2009.08.19)
% 数据是一个4579*6的double型的矩阵,每一行表示每一天的上证指数
% 6列分别表示当天上证指数的开盘指数,指数最高值,指数最低值,收盘指数,当日交易量,当日交易额.
load chapter_sh.mat;

% 提取数据
[m,n] = size(sh);
ts = sh(2:m,1);    % 选取2到4579个交易日内每日的开盘指数作为因变量
tsx =sh(1:m-1,:); %选取1到4578个交易日

% 画出原始上证指数的每日开盘数
figure;   % 1图
plot(ts,'LineWidth',2);
title('上证指数的每日开盘数(1990.12.20-2009.08.19)','FontSize',12);
xlabel('交易日天数(1990.12.19-2009.08.19)','FontSize',12);
ylabel('开盘数','FontSize',12);
grid on;  % 就是在画图的时候添加网络格线

% 数据预处理,将原始数据进行归一化
ts = ts';
tsx = tsx';

% mapminmax为matlab自带的映射函数	
% 对ts进行归一化
[TS,TSps] = mapminmax(ts,1,2);	%归一化在区间[1 2]

% 画出原始上证指数的每日开盘数归一化后的图像
figure;     % 2图
plot(TS,'LineWidth',2);
title('原始上证指数的每日开盘数归一化后的图像','FontSize',12);
xlabel('交易日天数(1990.12.19-2009.08.19)','FontSize',12);
ylabel('归一化后的开盘数','FontSize',12);
grid on;  % 就是在画图的时候添加网络格线
% 对TS进行转置,以符合libsvm工具箱的数据格式要求
TS = TS';

% mapminmax为matlab自带的映射函数
% 对tsx进行归一化
[TSX,TSXps] = mapminmax(tsx,1,2);	%归一化在区间[1 2]
% 对TSX进行转置,以符合libsvm工具箱的数据格式要求
TSX = TSX';

%% 选择回归预测分析最佳的SVM参数c&g

% 首先进行粗略选择: 
[bestmse,bestc,bestg] = SVMcgForRegress(TS,TSX,-8,8,-8,8);% 参数c和g

% 打印粗略选择结果
disp('打印粗略选择结果');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);%格式数据转换成字符串  Cross Validation交叉验证
disp(str);

% 根据粗略选择的结果图再进行精细选择: 
[bestmse,bestc,bestg] = SVMcgForRegress(TS,TSX,-4,4,-4,4,3,0.5,0.5,0.05);

% 打印精细选择结果
disp('打印精细选择结果');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

%% 利用回归预测分析最佳的参数进行SVM网络训练
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(TS,TSX,cmd);

%% SVM网络回归预测
[predict,mse,prob_estimates] = svmpredict(TS,TSX,model);%课本138页没有prob_estimates
predict = mapminmax('reverse',predict',TSps); %反归一化
predict = predict';

% 打印回归结果
str = sprintf( '均方误差 MSE = %g 相关系数 R = %g%%',mse(2),mse(3)*100);%句法为[s, errmsg] = sprintf(format, A, ...)可以把矩阵A做数据格式的转换，格式就是format参数。
disp(str);

%% 结果分析
figure;      % 7图
hold on;
plot(ts,'-o');
plot(predict,'r-^');
legend('原始数据','回归预测数据');
hold off;
title('原始数据和回归预测数据对比','FontSize',12);
xlabel('交易日天数(1990.12.19-2009.08.19)','FontSize',12);
ylabel('开盘数','FontSize',12);
grid on;

figure;    % 8图
error = predict - ts';
plot(error,'rd');
title('误差图(predicted data - original data)','FontSize',12);
xlabel('交易日天数(1990.12.19-2009.08.19)','FontSize',12);
ylabel('误差量','FontSize',12);
grid on;

figure;   % 9图
error = (predict - ts')./ts';
plot(error,'rd');
title('相对误差图(predicted data - original data)/original data','FontSize',12);
xlabel('交易日天数(1990.12.19-2009.08.19)','FontSize',12);
ylabel('相对误差量','FontSize',12);
grid on;
snapnow; %Force snapshot of image for inclusion in published document
toc; %toc函数会自动计算时间差

%% 子函数 SVMcgForRegress.m
function [mse,bestc,bestg] = SVMcgForRegress(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,msestep)
%SVMcg cross validation by faruto

%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2010.01.17
%Super Moderator @ www.ilovematlab.cn

% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

% about the parameters of SVMcg 
if nargin < 10
    msestep = 0.06;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end
% X:c Y:g cg:acc
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);%meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的。它也可以是更高维的。生成size(b)Xsize(a)大小的矩阵A和B。它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列
[m,n] = size(X);
cg = zeros(m,n);

eps = 10^(-4);

bestc = 0;
bestg = 0;
mse = Inf;%infinite的前三个字母，无穷大的意思。
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3 -p 0.1'];
        cg(i,j) = svmtrain(train_label, train, cmd);
        
        if cg(i,j) < mse
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
        if abs( cg(i,j)-mse )<=eps && bestc > basenum^X(i,j)
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
    end
end
% to draw the acc with different c & g
[cg,ps] = mapminmax(cg,0,1);
figure;
[C,h] = contour(X,Y,cg,0:msestep:0.5);
clabel(C,h,'FontSize',10,'Color','r');
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
firstline = 'SVR参数选择结果图(等高线图)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVmse=',num2str(mse)];
title({firstline;secondline},'Fontsize',12);
grid on;

figure;
meshc(X,Y,cg);
% mesh(X,Y,cg);
% surf(X,Y,cg);
axis([cmin,cmax,gmin,gmax,0,1]);
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('MSE','FontSize',12);
firstline = 'SVR参数选择结果图(3D视图)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVmse=',num2str(mse)];
title({firstline;secondline},'Fontsize',12);

