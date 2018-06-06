tic;
clear all;
close all;
load  x110.mat 
xy=x110(1:1100);
K=x110(1:1398);
x1=K;
N=length(x1); 
L=500;
[y,r,vr]=ssa(x1,L);
global y;
x3=y(1:1100);
n=length(x3);
for i=1:6
    sample(i,:)=x3(i:i+n-6);   %构造train
end
input=sample(1:5,:);
output=sample(6,:);
input_train=input(:,1:1095);
output_train=output(1:1095);

[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
inputn=inputn';
outputn=outputn';
for i=1:288
    p_test=y(i+1100:i+1104);
end
for i=1:288
     Y(:,i)=x110(1105+i);%一步
end
[inputn1,inputps]=mapminmax(p_test);
P=inputn1';
[outputn1,outputps]=mapminmax(Y)
P1=outputn1;

Tol=1.0e-4;  
n=25;%鸟巢个数
% Discovery rate of alien eggs/solutions
pa=0.25;

                                                          %为最大迭代次数限制
%% Simple bounds of the search domain
% Lower bounds
nd=2; 
Lb=0.01*ones(1,nd); 
% Upper bounds
Ub=100*ones(1,nd);                                                              %随机产生初始解
% Random initial solutions
for i=1:n,      
nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
end
%得到当前的最优解
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

N_iter=0;                                                                   %开始迭代
%% Starting iterations
while N_iter<400
       
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
      N_iter=N_iter+n
    % Find the best objective so far  
    if fnew<fmin,
        fmin=fnew;
        bestnest=best;
    end
end %% End of iterations(迭代)



bestc=bestnest(1);
bestg=bestnest(2);

cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(outputn,inputn,cmd);

%% SVM网络回归预测
[predict0,mse,~] = svmpredict(outputn,inputn,model);
predict0 = mapminmax('reverse',predict0',outputps);
predict0 = predict0';


for i=1:288 
[predict(1,i),mse,prob_estimates] = svmpredict(P1,P,model);
 yc(1,i)=mapminmax('reverse',predict(1,i),outputps);
end
error1=yc-Y;
mape1=mean(abs(error1)./Y);
MAPE1=mean(mape1)*100;  
mae1=mean(abs(error1));
MAE1=mean(mae1);
rmse1=sqrt(mean(error1.^2));
RMSE1=mean(rmse1);
W1=[MAE1 MAPE1 RMSE1]'