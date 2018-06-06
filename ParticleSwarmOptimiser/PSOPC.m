function [fbestval,bestparticle] = PSO(NDim,ubound,lbound, MaxIter, fname)

% function [fbestval,bestparticle] = PSOPC(fname,bound,vmax,NDim,MaxIter)
%                          
%   Run a PSO with Passive Congregation (PSOPC) algorithm
%   
%Input Arguments:
%   fname       - the name of the evaluation .m function
%   NDim        - dimension of the evalation function
%   MaxIter     - maximum iteration


% Modified Particle Swarm Optimization for Matlab  
% Copyright (C) 2003 Shan He, the University of Liverpool
% Intelligence Engineering & Automation Group
% Last modifed 13-Aug-03

ploton=0;
figure

flag=0;
iteration = 0;
%MaxIter =1000;  % maximum number of iteration
PopSize=20;     % population of particles
%NDim=10;         % Number of dimension of search space
c1 = .6;       % PSO parameter C1
c2 = .6;       % PSO parameter C2
w=0.8;          % Inertia weigth
% decrease the inertia 
startwaight = 0.9;
endwaight = 0.5;
waightstep = (startwaight-endwaight)/MaxIter;

c3step = (1 - 0.6)/MaxIter;


% Defined lower bound and upper bound.
LowerBound = zeros(NDim,PopSize);
UpperBound = zeros(NDim,PopSize);
for i=1:PopSize
    LowerBound(:,i) = lbound';
    UpperBound(:,i) = ubound';
end


DResult = 0;    % Desired results
population =  rand(NDim, PopSize).*(UpperBound-LowerBound) + LowerBound;     % Initialize swarm population
vmax = ones(NDim,PopSize);

for i=1:NDim
    vmax(i,:)=(UpperBound(i,:)-LowerBound(i,:))/2;
end
velocity = vmax.*rand(1);      % Initialize velocity

exexutefunction=strcat(fname,'(population(:,i))');
% Evaluate initial population
for i = 1:PopSize,
    fvalue(i) = eval(exexutefunction);
end

pbest = population;   % Initializing Best positions¡¯ matrix
fpbest = fvalue;      % Initializing the corresponding function values
% Finding best particle in initial population
[fbestval,index] = min(fvalue);    % Find the globe best   
[fsortval, sortindex] = sort(fvalue);
while(flag == 0) & (iteration < MaxIter)
    iteration = iteration +1;
    w = startwaight - iteration*waightstep;
    for i=1:PopSize
        gbest(:,i) = population(:,index);
    end
    
    for i=1:PopSize
        rparticle(:,i) = population(:,floor(PopSize*rand(1))+1);
    end    
    
    R1 = rand(NDim, PopSize);
    R2 = rand(NDim, PopSize);
    R3 = rand(NDim, PopSize);
    
    c3 = 0.6 + c3step*iteration;
    stationary=ones(NDim, PopSize);
    stationary(:,index)=0;
    sortmatrix = repmat(sortindex, NDim, 1)./PopSize;
    velocity = w*velocity + c1*R1.*(pbest-population) + c2*R2.*(gbest-population) + c3*R3.*sortmatrix.*(rparticle-population);
    % Update the swarm particle
    population = population + velocity;
    
    
    % Prevent particles from flying outside search space
    %OutFlag = population<=LowerBound | population>=UpperBound;
    %population = population - OutFlag.*velocity;
    
    % Evaluate the new swarm
    for i = 1:PopSize,
        fvalue(i) = eval(exexutefunction);
    end
    % Updating the pbest for each particle
    changeColumns = fvalue < fpbest;
    pbest(:, find(changeColumns)) = population(:, find(changeColumns));     % find(changeColumns) find the columns which the values are 1
    fpbest = fpbest.*( ~changeColumns) + fvalue.*changeColumns;             % update fpbest value if fvalue is less than fpbest

        
    % Updating index 
    [fbestval, index] = min(fvalue);
    [fsortval, sortindex] = sort(fvalue);
    
     % plot best fitness
     %hold on;
     Best(iteration) =fbestval;
     semilogy(Best,'r--');xlabel('generation'); ylabel('f(x)');
     text(0.5,0.95,['Best = ', num2str(Best(iteration))],'Units','normalized'); 
     drawnow;
end  

bestparticle = population(:,index)
 

