function [bestCVmse,bestc,pso_option] = psoRVMcgForRegress(train_label,train,pso_option)
% psoSVMcgForClass
%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto
%last modified 2011.06.08
% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion
% a toolbox with implements for support vector machines based on libsvm, 2011.
% Software available at http://www.matlabsky.com
%
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% 参数初始化
% if nargin == 2
%     pso_option = struct('c1',1.5,'c2',1.7,'maxgen',200,'sizepop',20, ...
%                     'k',0.6,'wV',1,'wP',1,'v',5, ...
%                      'popcmax',10^2,'popcmin',10^(-1),'popgmax',10^3,'popgmin',10^(-2));
% end
% c1:初始为1.5,pso参数局部搜索能力
% c2:初始为1.7,pso参数全局搜索能力
% maxgen:初始为200,最大进化数量
% sizepop:初始为20,种群最大数量
% k:初始为0.6(k belongs to [0.1,1.0]),速率和x的关系(V = kX)
% wV:初始为1(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
% wP:初始为1,种群更新公式中速度前面的弹性系数
% v:初始为5,SVM Cross Validation参数
% popcmax:初始为100,SVM 参数c的变化的最大值.
% popcmin:初始为0.1,SVM 参数c的变化的最小值.
% popgmax:初始为1000,SVM 参数g的变化的最大值.
% popgmin:初始为0.01,SVM 参数c的变化的最小值.

Vcmax = pso_option.k*pso_option.popcmax;
Vcmin = -Vcmax ;

eps = 10^(-4);

%% 产生初始粒子和速度
for i=1:pso_option.sizepop
    
    % 随机产生种群和速度
    pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
    V(i,1)=Vcmax*rands(1,1);
    
    % 计算初始适应度
    [used,weights,bias] = RVM_training(train,train_label,pop(i,1));
    % Evaluate the model over the training data
    kernel= 'gauss';
    PHI	= SB1_KernelFunction(train,train,kernel,pop(i,1));
    train_label_rvm = PHI(:,used)*weights + bias;
    % MAPE作为适应度函数
    fitness(i)=errperf(train_label,train_label_rvm,'mape');
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点
local_x=pop;    % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,pso_option.maxgen);

%% 迭代寻优
for i=1:pso_option.maxgen
    
    for j=1:pso_option.sizepop
        
        %速度更新
        V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        if V(j,1) > Vcmax
            V(j,1) = Vcmax;
        end
        if V(j,1) < Vcmin
            V(j,1) = Vcmin;
        end
        
        %种群更新
        pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
        if pop(j,1) > pso_option.popcmax
            pop(j,1) = pso_option.popcmax;
        end
        if pop(j,1) < pso_option.popcmin
            pop(j,1) = pso_option.popcmin;
        end
        
        % 自适应粒子变异
%         if rand>0.5
%             k=ceil(2*rand);
%             if k == 1
%                 pop(j,k) = (20-1)*rand+1;
%             end
%         end
        
        %适应度值
        [used,weights,bias] = RVM_training(train,train_label,pop(j,1));
        % Evaluate the model over the training data
        kernel= 'gauss';
        PHI	= SB1_KernelFunction(train,train,kernel,pop(j,1));
        train_label_rvm = PHI(:,used)*weights + bias;
        % MAPE作为适应度函数
        fitness(j)=errperf(train_label,train_label_rvm,'mape');
        
        %个体最优更新
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        %群体最优更新
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
        
        if abs( fitness(j)-global_fitness )<=eps && pop(j,1) < global_x(1)
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
        
    end
    
    fit_gen(i)=global_fitness;
    avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
end

%% 结果分析
figure;
hold on;
semilogy(fit_gen,'r*-','LineWidth',1.5);
semilogy(avgfitness_gen,'o-','LineWidth',1.5);
legend('Best fitness','Average fitness');
xlabel('No. of Generation'); % ,'FontSize',12
ylabel('Best fitness obtained so far');
grid on;

bestc = global_x(1);

bestCVmse = fit_gen(pso_option.maxgen);

% line1 = '适应度曲线MSE[PSOmethod]';
% line2 = ['(参数c1=',num2str(pso_option.c1), ...
%     ',c2=',num2str(pso_option.c2),',终止代数=', ...
%     num2str(pso_option.maxgen),',种群数量pop=', ...
%     num2str(pso_option.sizepop),')'];
% line3 = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVmse=',num2str(bestCVmse),];
% title({line1;line2;line3},'FontSize',12);
line1 = 'Fitness curve of PSO';
% line2 = ['(终止代数=', ...
%     num2str(gen),',种群数量pop=', ...
%     num2str(NIND),')'];
% line3 = ['Best c=',num2str(Bestc),' g=',num2str(Bestg), ...
%     ' MSE=',num2str(BestMSE)];
% title({line1;line2;line3},'FontSize',12);
title({line1});


