
%% ���ݵ���ȡ��Ԥ����                            

% �������������ָ֤��(1990.12.19-2009.08.19)
% ������һ��4579*6��double�͵ľ���,ÿһ�б�ʾÿһ�����ָ֤��
% 6�зֱ��ʾ������ָ֤���Ŀ���ָ��,ָ�����ֵ,ָ�����ֵ,����ָ��,���ս�����,���ս��׶�.
clear
clc
load chapter_sh.mat;

% ��ȡ����
[m,n] = size(sh);
ts = sh(2:m,1);    % ѡȡ2��4579����������ÿ�յĿ���ָ����Ϊ�����
tsx =sh(1:m-1,:); %ѡȡ1��4578��������

% ����Ԥ����,��ԭʼ���ݽ��й�һ��
ts = ts';
tsx = tsx';

% mapminmaxΪmatlab�Դ���ӳ�亯��	
% ��ts���й�һ��
[TS,TSps] = mapminmax(ts,1,2);	%��һ��������[1 2]
% ��TSX����ת��,�Է���libsvm����������ݸ�ʽҪ��
TS = TS';

% mapminmaxΪmatlab�Դ���ӳ�亯��
% ��tsx���й�һ��
[TSX,TSXps] = mapminmax(tsx,1,2);	%��һ��������[1 2]
% ��TSX����ת��,�Է���libsvm����������ݸ�ʽҪ��
TSX = TSX';

Tol=1.0e-5;  
n=25;%�񳲸���
% Discovery rate of alien eggs/solutions
pa=0.25;

                                                          %Ϊ��������������
%% Simple bounds of the search domain
% Lower bounds
nd=2; 
Lb=0.01*ones(1,nd); 
% Upper bounds
Ub=100*ones(1,nd);                                                              %���������ʼ��
% Random initial solutions
for i=1:n,      
nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
end
%�õ���ǰ�����Ž�
% Get the current best
for i=1:n
    fitness(i)=fun(nest(i,:));
end

fitness=10^10*ones(n,1);
[fmin,bestnest,nest,fitness]=get_best_nest(nest,nest,fitness,Ub,Lb);
for i=1:n
    nest(i,find(nest(i,:)>Ub(1)))=Ub(1);
    nest(i,find(nest(i,:)<Lb(1)))=Lb(1);
end

N_iter=0;                                                                   %��ʼ����
%% Starting iterations
for iter=1:1 %while (fmin>Tol),

    % Generate new solutions (but keep the current best)
     new_nest=get_cuckoos(nest,bestnest,Lb,Ub);   
     [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness,Ub,Lb);
    % Update the counter
      N_iter=N_iter+n; 
    % Discovery and randomization
      new_nest=empty_nests(nest,Lb,Ub,pa) ;
    
    % Evaluate this set of solutions
      [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness,Ub,Lb);
    % Update the counter again
      N_iter=N_iter+n;
    % Find the best objective so far  
    if fnew<fmin,
        fmin=fnew;
        bestnest=best;
    end
end %% End of iterations(����)



bestc=bestnest(1);
bestg=bestnest(2);

cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(TS,TSX,cmd);

%% SVM����ع�Ԥ��
[predict,mse,~] = svmpredict(TS,TSX,model);%�α�138ҳû��prob_estimates
predict = mapminmax('reverse',predict',TSps); %����һ��
predict = predict';
error=predict-ts';
errorn=sum(abs(error));
figure;     
hold on;
plot(ts,'-o');
plot(predict,'r-^');
legend('ԭʼ����','�ع�Ԥ������');
hold off;
title('ԭʼ���ݺͻع�Ԥ�����ݶԱ�','FontSize',12);
xlabel('����������(1990.12.19-2009.08.19)','FontSize',12);
ylabel('������','FontSize',12);
grid on;
