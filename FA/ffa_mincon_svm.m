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

% �㷨������ʼ Start FA
function [nbest,fbest]=ffa_mincon_svm(costfhandle,u0, Lb, Ub, para,train_wine_labels,train_wine,test_wine_labels,test_wine)
% ���������� Check input parameters (otherwise set as default values)
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

% ����Ͻ��������½����������Ƿ���ͬ Check if the upper bound & lower bound are the same size
if length(Lb) ~=length(Ub)
    disp('Simple bounds/limits are improper!')
    return
end

% ������Ż�����ά�� Calcualte dimension
d=length(u0);

% ��ʼ��Ŀ�꺯��ֵ Initial values of an array
zn=ones(n,1)*10^100;
% ------------------------------------------------
% ��ʼ��ө���λ�� generating the initial locations of n fireflies
[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0);

for k=1:MaxGeneration % ������ʼ
% ����alpha����ѡ��This line of reducing alpha is optional
 alpha=alpha_new(alpha,MaxGeneration);

% ��ÿ��ө������Ŀ�꺯��ֵ Evaluate new solutions (for all n fireflies)
for i=1:n
    zn(i)=costfhandle(ns(i,:),train_wine_labels,train_wine,test_wine_labels,test_wine);
    Lightn(i)=zn(i);
end

% ������������ Ranking fireflies by their light intensity/objectives
[Lightn,Index]=sort(zn);
ns_tmp=ns;
for i=1:n
    ns(i,:)=ns_tmp(Index(i),:);
end

%% �ҳ���ǰ���� Find the current best
nso=ns;
Lighto=Lightn;
nbest=ns(1,:);
Lightbest=Lightn(1);

% �������ֵ For output only
fbest=Lightbest;

% ����ŷ����ƶ� Move all fireflies to the better locations
[ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,alpha,betamin,gamma,Lb,Ub);
end

% ----- All the subfunctions are listed here ------------
% ��ʼ��ө���λ�� The initial locations of n fireflies
function [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)
ns=zeros(n,d);
if ~isempty(Lb) % ����������޲�Ϊ�� if there are bounds/limits
   for i=1:n
       ns(i,:)=Lb+(Ub-Lb).*rand(1,d); % ����ȡֵ��Χ�����ȡֵ
   end
else % ���û�����ò�������
    for i=1:n
        ns(i,:)=u0+randn(1,d); % ��ԭ�в����ϼӰ�����
    end
end
% ��ʼ��Ŀ�꺯�� initial value before function evaluations
Lightn=ones(n,1)*10^100;

% Move all fireflies toward brighter ones
function [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,alpha,betamin,gamma,Lb,Ub)
% ����ȡֵ��Χ����ֵ Scaling of the system
scale=abs(Ub-Lb);

% ����ө��� Updating fireflies
for i=1:n
    % The attractiveness parameter beta=exp(-gamma*r)
    for j=1:n
        r=sqrt(sum((ns(i,:)-ns(j,:)).^2));
        % Update moves
        if Lightn(i)>Lighto(j) % ���i��j���ȸ�ǿ Brighter and more attractive
            beta0=1;
            beta=(beta0-betamin)*exp(-gamma*r.^2)+betamin;
            tmpf=alpha.*(rand(1,d)-0.5).*scale;
            ns(i,:)=ns(i,:).*(1-beta)+nso(j,:).*beta+tmpf;
        end
    end % end for j
end % end for i

% ��ֹԽ�� Check if the updated solutions/locations are within limits
[ns]=findlimits(n,ns,Lb,Ub);

% This function is optional, as it is not in the original FA
% The idea to reduce randomness is to increase the convergence,
% however, if you reduce randomness too quickly, then premature
% convergence can occur. So use with care.
% alpha�������º��� 
function alpha=alpha_new(alpha,NGen)
% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
% alpha_0=0.9
delta=1-(10^(-4)/0.9)^(1/NGen);
alpha=(1-delta)*alpha;

% ��ֹԽ�� Make sure the fireflies are within the bounds/limits
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