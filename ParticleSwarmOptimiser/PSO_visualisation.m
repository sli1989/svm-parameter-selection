function [fbestval,bestparticle] = PSO_visualisation(fname,NDim,MaxIter)

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

PopSize=10;     % population of particles
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

%% Plot 3D function
close all
[X,Y] = meshgrid(Bound(:,1):1:Bound(:,2));
[xx yy] = size(X);
for ii=1:xx
    for jj=1:yy
        Z(ii,jj) = feval(fname, [X(ii,jj); Y(ii,jj)]);
    end
end
mesh(X,Y,Z)
hold on
plot3(0,0,0,'rx', 'LineWidth',2,'MarkerSize',8)


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

previous_pop = population;


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
    gbest = repmat(population(:,index), 1, length(population));

    % Update velocity
    velocity = w*(velocity + c1*R1.*(pbest-population) + c2*R2.*(gbest-population));

    % Clamped to the maximum velocity
    velocity=(velocity<-vmax).*(-vmax)+(velocity>vmax).*(vmax)+(velocity>-vmax&velocity<vmax).*velocity;

    % Update the swarm particle
    population = population + velocity;

    % Prevent particles from flying outside search space
    population(population>UpperBound)=UpperBound(population>UpperBound);                % Clamped to upper range
    population(population<LowerBound)=LowerBound(population<LowerBound);                % Clamped to lower range

    % plot best fitness
    Best(iteration) = fbestval;
    fprintf(1,'%e   ',fbestval);
    if iteration/5==floor(iteration/5)
        fprintf(1,'\n');
    end
    

    %% Plot 3D visulisation
    hold on;
    for jj=1:PopSize
        position = population(:,jj);
        z = feval(fname,position);
        
        pre_pos = previous_pop(:,jj);
        pre_z = feval(fname,pre_pos);
        plot3(pre_pos(1), pre_pos(2), pre_z,'kx', 'LineWidth',2,'MarkerSize',8); 
        plot3(position(1), position(2), z,'o','Color',[mod(iteration,2),iteration/MaxIter, mod(iteration+1,2)], 'LineWidth',2,'MarkerSize',8);     
        plot3([pre_pos(1),position(1)], [pre_pos(2),position(2)], [pre_z,z], '-.k');     

    end

    previous_pop = population;

end




