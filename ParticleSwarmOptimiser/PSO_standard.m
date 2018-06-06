function [fbestval,bestparticle] = PSOnewstandardnew2(fname,NDim,MaxIter)

% function [fbestval,bestparticle] = PSOnewOct28(fname,bound,vmax,NDim,MaxIter)
%
%   Run a PSO algorithm
%
%Input Arguments:
%   fname       - the name of the evaluation .m function
%   NDim        - dimension of the evalation function
%   MaxIter     - maximum iteration


% Standard version of Particle Swarm Optimization for Matlab
% Copyright (C) 2002 Shan He, the University of Liverpool
% Intelligence Engineering & Automation Group

%% Parameters
ploton=0;
flag=0;
iteration = 0;

PopSize=50;     % population of particles
w=0.73;
c1=2.05;
c2=2.05;

%% Lower/Upper bounds
Bound=eval(fname);
% Defined lower bound and upper bound.
LowerBound = zeros(NDim,PopSize);
UpperBound = zeros(NDim,PopSize);
for i=1:PopSize
    LowerBound(:,i)=Bound(:,1);
    UpperBound(:,i)=Bound(:,2);
end

%% Initialize swarm population randomly
population =  rand(NDim, PopSize).*(UpperBound-LowerBound) + LowerBound;     

%% Randomly initialise velocity
vmax = ones(NDim,PopSize);
for i=1:NDim
    vmax(i,:)=(UpperBound(i,:)-LowerBound(i,:))/10;
end
velocity=vmax.*rand(NDim,PopSize);


%% Evaluate initial population
exexutefunction=strcat(fname,'(population(:,i))');
for i = 1:PopSize,
    fvalue(i) = eval(exexutefunction);
end

pbest = population;   % Initializing Best positions matrix
fpbest = fvalue;      % Initializing the corresponding function values
% Finding best particle in initial population
[fbestval,index] = min(fvalue);    % Find the globe best


%% Main loop
while(flag == 0) & (iteration < MaxIter)
    iteration = iteration +1;


    R1 = rand(NDim, PopSize);
    R2 = rand(NDim, PopSize);


    % Evaluate the new swarm
    for i = 1:PopSize,
        fvalue(i) = eval(exexutefunction);
    end

    % Updating the pbest for each particle
    changeColumns = fvalue < fpbest;
    pbest(:, find(changeColumns)) = population(:, find(changeColumns));     % find(changeColumns) find the columns which the values are 1
    fpbest = fpbest.*( ~changeColumns) + fvalue.*changeColumns;

    % Updating best particle gbest
    [fbestval, index] = min(fpbest);
    for i=1:PopSize
        gbest(:,i) = population(:,index);
    end

    velocity = w*(velocity + c1*R1.*(pbest-population) + c2*R2.*(gbest-population));

    % Clip the maximum velocity
    velocity=(velocity<-vmax).*(-vmax)+(velocity>vmax).*(vmax)+(velocity>-vmax&velocity<vmax).*velocity;


    % Update the swarm particle
    population = population + velocity;

    % Prevent particles from flying outside search space
    population(population>UpperBound)=UpperBound(population>UpperBound);                % crop to upper range
    population(population<LowerBound)=LowerBound(population<LowerBound);                % crop to lower range

    % plot best fitness
    Best(iteration) =fbestval;
    fprintf(1,'%e   ',fbestval);
    if iteration/5==floor(iteration/5)
        fprintf(1,'\n');
    end


end


filename=strcat('G:\GSOdata\PSOconstriction2-',fname);
filename=strcat(filename, '.txt');
fid = fopen(filename,'a');
fprintf(fid,'%e\n',Best);
fclose(fid);
