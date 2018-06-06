%%% End of the part to be modified -------------------%%%
%%% --------------------------------------------------%%%
%%% Do not modify the following codes unless you want %%%
%%% to improve its performance etc                    %%%
% -------------------------------------------------------
% ===Start of the Firefly Algorithm Implementation ======
%         Lb = lower bounds/limits
%         Ub = upper bounds/limits
%   para == optional (to control the Firefly algorithm)
% Outputs: nbest   = the best solution found so far
%          fbest   = the best objective value
%      NumEval = number of evaluations: n*MaxGeneration
% Optional:
% The alpha can be reduced (as to reduce the randomness)
% ---------------------------------------------------------

% 算法主程序开始 Start FA
function [nbest,fbest]=ffa_mincon_svm(costfhandle,u0, Lb, Ub, para,train_wine_labels,train_wine,test_wine_labels,test_wine)
% 检查输入参数 Check input parameters (otherwise set as default values)
if nargin<5
    para=[20 100 0.25 0.20 1];
end
if nargin<4
    Ub=[];
end
if nargin<3
    Lb=[];
end
if nargin<2
    disp('Usuage: FA_mincon(@cost,u0,Lb,Ub,para)');
end

% n=number of fireflies
% MaxGeneration=number of pseudo time steps
% ------------------------------------------------
% alpha=0.25;      % Randomness 0--1 (highly random)
% betamn=0.20;     % minimum value of beta
% gamma=1;         % Absorption coefficient
% ------------------------------------------------
n=para(1);
MaxGeneration=para(2);
alpha=para(3);
betamin=para(4);
gamma=para(5);

% 检查上界向量与下界向量长度是否相同 Check if the upper bound & lower bound are the same size
if length(Lb) ~=length(Ub)
    disp('Simple bounds/limits are improper!')
    return
end

% 计算待优化参数维度 Calcualte dimension
d=length(u0);

% 初始化目标函数值 Initial values of an array
zn=ones(n,1)*10^100;
% ------------------------------------------------
% 初始化萤火虫位置 generating the initial locations of n fireflies
[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0);

for k=1:MaxGeneration % 迭代开始
% 更新alpha（可选）This line of reducing alpha is optional
 alpha=alpha_new(alpha,MaxGeneration);

% 对每个萤火虫计算目标函数值 Evaluate new solutions (for all n fireflies)
for i=1:n
    zn(i)=costfhandle(ns(i,:),train_wine_labels,train_wine,test_wine_labels,test_wine);
    Lightn(i)=zn(i);
end

% 根据亮度排序 Ranking fireflies by their light intensity/objectives
[Lightn,Index]=sort(zn);
ns_tmp=ns;
for i=1:n
    ns(i,:)=ns_tmp(Index(i),:);
end

%% 找出当前最优 Find the current best
nso=ns;
Lighto=Lightn;
nbest=ns(1,:);
Lightbest=Lightn(1);

% 另存最优值 For output only
fbest=Lightbest;

% 向较优方向移动 Move all fireflies to the better locations
[ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,alpha,betamin,gamma,Lb,Ub);
end

% ----- All the subfunctions are listed here ------------
% 初始化萤火虫位置 The initial locations of n fireflies
function [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)
ns=zeros(n,d);
if ~isempty(Lb) % 如果参数界限不为空 if there are bounds/limits
   for i=1:n
       ns(i,:)=Lb+(Ub-Lb).*rand(1,d); % 则在取值范围内随机取值
   end
else % 如果没有设置参数界限
    for i=1:n
        ns(i,:)=u0+randn(1,d); % 在原有参数上加白噪声
    end
end
% 初始化目标函数 initial value before function evaluations
Lightn=ones(n,1)*10^100;

% Move all fireflies toward brighter ones
function [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,alpha,betamin,gamma,Lb,Ub)
% 参数取值范围绝对值 Scaling of the system
scale=abs(Ub-Lb);

% 更新萤火虫 Updating fireflies
for i=1:n
    % The attractiveness parameter beta=exp(-gamma*r)
    for j=1:n
        r=sqrt(sum((ns(i,:)-ns(j,:)).^2));
        % Update moves
        if Lightn(i)>Lighto(j) % 如果i比j亮度更强 Brighter and more attractive
            beta0=1;
            beta=(beta0-betamin)*exp(-gamma*r.^2)+betamin;
            tmpf=alpha.*(rand(1,d)-0.5).*scale;
            ns(i,:)=ns(i,:).*(1-beta)+nso(j,:).*beta+tmpf;
        end
    end % end for j
end % end for i

% 防止越界 Check if the updated solutions/locations are within limits
[ns]=findlimits(n,ns,Lb,Ub);

% This function is optional, as it is not in the original FA
% The idea to reduce randomness is to increase the convergence,
% however, if you reduce randomness too quickly, then premature
% convergence can occur. So use with care.
% alpha参数更新函数 
function alpha=alpha_new(alpha,NGen)
% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
% alpha_0=0.9
delta=1-(10^(-4)/0.9)^(1/NGen);
alpha=(1-delta)*alpha;

% 防止越界 Make sure the fireflies are within the bounds/limits
function [ns]=findlimits(n,ns,Lb,Ub)
for i=1:n
    % Apply the lower bound
    ns_tmp=ns(i,:);
    I=ns_tmp<Lb;
    ns_tmp(I)=Lb(I);
    % Apply the upper bounds
    J=ns_tmp>Ub;
    ns_tmp(J)=Ub(J);
    % Update this new move
    ns(i,:)=ns_tmp;
end