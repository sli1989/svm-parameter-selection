function [BestMSE,Bestc,ga_option] = gaRVMcgForRegress(train_label,train_data,ga_option)
% gaSVMcgForClass
%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto
%last modified 2011.06.08
%
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
%     ga_option = struct('maxgen',200,'sizepop',20,'ggap',0.9,...
%         'cbound',[0,100],'gbound',[0,1000],'v',5);
% end

% maxgen:最大的进化代数,默认为200,一般取值范围为[100,500]
% sizepop:种群最大数量,默认为20,一般取值范围为[20,100]
% cbound = [cmin,cmax],参数c的变化范围,默认为(0,100]
% gbound = [gmin,gmax],参数g的变化范围,默认为[0,1000]
% v:SVM Cross Validation参数,默认为5

%%
MAXGEN = ga_option.maxgen;
NIND = ga_option.sizepop;
NVAR = 1; %个数长度
PRECI = 20;% 个体二进制编码位数
GGAP = ga_option.ggap; %代沟部分种群被复制
trace = zeros(MAXGEN,2);

FieldID = ...
    [rep([PRECI],[1,NVAR]);[ga_option.cbound(1);ga_option.cbound(2)];[1;0;0;1]];

Chrom = crtbp(NIND,NVAR*PRECI);

gen = 1;
v = ga_option.v;
BestMSE = inf;
Bestc = 0;

%%
cg = bs2rv(Chrom,FieldID);
% train_data=x_rvm;
% train_label=y_rvm;
for nind = 1:NIND
    [used,weights,bias] = RVM_training(train_data,train_label,cg(nind,1));
    % Evaluate the model over the training data
    kernel= 'gauss';
    PHI	= SB1_KernelFunction(train_data,train_data,kernel,cg(nind,1));
    train_label_rvm = PHI(:,used)*weights + bias;
    % MAPE作为适应度函数
    ObjV(nind,1)=errperf(train_label,train_label_rvm,'mape');
end
[BestMSE,I] = min(ObjV);
Bestc = cg(I,1);
%%
while 1
    FitnV = ranking(ObjV);
    
    SelCh = select('sus',Chrom,FitnV,GGAP);
    SelCh = recombin('xovsp',SelCh,0.7);
    SelCh = mut(SelCh);
    
    cg = bs2rv(SelCh,FieldID);
    for nind = 1:size(SelCh,1)
        [used,weights,bias] = RVM_training(train_data,train_label,cg(nind,1));
        % Evaluate the model over the training data
        kernel= 'gauss';
        PHI	= SB1_KernelFunction(train_data,train_data,kernel,cg(nind,1));
        train_label_rvm = PHI(:,used)*weights + bias;
        % MAPE作为适应度函数
        ObjVSel(nind,1)=errperf(train_label,train_label_rvm,'mape');
    end
    
    [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);
    
    [NewBestCVaccuracy,I] = min(ObjV);
    cg_temp = bs2rv(Chrom,FieldID);
    temp_NewBestCVaccuracy = NewBestCVaccuracy;
    
    if NewBestCVaccuracy < BestMSE
        BestMSE = NewBestCVaccuracy;
        Bestc = cg_temp(I,1);
    end
    
    if abs( NewBestCVaccuracy-BestMSE ) <= 10^(-4) && ...
            cg_temp(I,1) < Bestc
        BestMSE = NewBestCVaccuracy;
        Bestc = cg_temp(I,1);
    end
    
    trace(gen,1) = min(ObjV);
    trace(gen,2) = sum(ObjV)/length(ObjV);
    
    if gen >= MAXGEN/2 && ...
            ( temp_NewBestCVaccuracy-BestMSE ) <= 10^(-4)
        break;
    end
    if gen == MAXGEN
        break;
    end
    gen = gen + 1;
end

%%
figure;
hold on;
trace = round(trace*10000)/10000;
% plot(trace(1:gen,1),'r*-','LineWidth',1.5);
% plot(trace(1:gen,2),'o-','LineWidth',1.5);
semilogy(trace(1:gen,1),'r*-','LineWidth',1.5);
semilogy(trace(1:gen,2),'o-','LineWidth',1.5);
legend('Best fitness','Average fitness');
xlabel('No. of Generation'); % ,'FontSize',12
ylabel('Best fitness obtained so far');
grid on;
axis auto;

line1 = 'Fitness curve of GA';
% line2 = ['(终止代数=', ...
%     num2str(gen),',种群数量pop=', ...
%     num2str(NIND),')'];
% line3 = ['Best c=',num2str(Bestc),' g=',num2str(Bestg), ...
%     ' MSE=',num2str(BestMSE)];
% title({line1;line2;line3},'FontSize',12);
title({line1});

% figure;
% [X,Y] = meshgrid(-10:0.5:10,-10:0.5:10);
% meshc(X,Y,trace(1:gen,1));
% % mesh(X,Y,cg);
% % surf(X,Y,cg);
% % axis([cmin,cmax,gmin,gmax,0,1]);
% xlabel('log2c','FontSize',12);
% ylabel('log2g','FontSize',12);
% zlabel('MSE','FontSize',12);
% firstline = 'SVR参数选择结果图(3D视图) GA';
% % secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
% %     ' CVmse=',num2str(mse)];
% % title({firstline;secondline},'Fontsize',12);
% title({firstline})
