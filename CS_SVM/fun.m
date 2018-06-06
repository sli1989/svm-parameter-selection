function errorn=fun(bestnest)
load  x110.mat 
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


bestc=bestnest(1);
bestg=bestnest(2);

cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(outputn,inputn,cmd);

%% SVM网络回归预测
[predict0,mse,~] = svmpredict(outputn,inputn,model);
predict0 = mapminmax('reverse',predict0',outputps);
predict0 = predict0';


%% 拟合结果分析
error0=output_train-predict0';
errorn=sum(abs(error0));

