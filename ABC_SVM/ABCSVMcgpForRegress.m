function [BestMSE,BestParams,ABCOpts] = ABCSVMcgpForRegress(train_label,train_data,ABCOpts)

% 若转载请注明：
% a toolbox with implements for support vector machines based on libsvm, 2011.
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% 参数初始化
if nargin == 2
    ABCOpts = struct( 'ColonySize',  20, ...   % Number of Employed Bees+ Number of Onlooker Bees
        'MaxCycles', 100,...   % Maximum cycle number in order to terminate the algorithm
        'ErrGoal',   1e-20, ...  % Error goal in order to terminate the algorithm (not used in the code in current version)
        'Dim',       3, ... % Number of parameters of the objective function
        'Limit',   50, ... % Control paramter in order to abandone the food source
        'lb',  [2^-8,2^-8,0.01],... % Lower bound of the parameters to be optimized
        'ub',  [2^8,2^8,1],... %Upper bound of the parameters to be optimized
        'v',3,...
        'RunTime',1); % Number of the runs
end
FoodNumber=ABCOpts.ColonySize/2;%The number of food sources equals the half of the colony size

GlobalMins=zeros(1,ABCOpts.RunTime);

for r=1:ABCOpts.RunTime
    
    % /*All food sources are initialized */
    %/*Variables are initialized in the range [lb,ub]. If each parameter has different range, use arrays lb[j], ub[j] instead of lb and ub */
    
    Range = repmat((ABCOpts.ub-ABCOpts.lb),[FoodNumber 1]);
    Lower = repmat(ABCOpts.lb, [FoodNumber 1]);
    Foods = rand(FoodNumber,ABCOpts.Dim) .* Range + Lower;
    
    for nind = 1:FoodNumber
        cmd = ['-v ',num2str(ABCOpts.v),' -c ',num2str(Foods(nind,1)),' -g ',num2str(Foods(nind,2)),' -p ',num2str(Foods(nind,3)),' -s 3'];
        ObjVal(nind,1) = svmtrain(train_label,train_data,cmd);
    end
    Fitness=calculateFitness(ObjVal);
    % ObjVal=feval(objfun,Foods);
    % Fitness=calculateFitness(ObjVal);
    
    %reset trial counters
    trial=zeros(1,FoodNumber);
    
    %/*The best food source is memorized*/
    BestInd=find(ObjVal==min(ObjVal));
    BestInd=BestInd(end);
    GlobalMin=ObjVal(BestInd);
    GlobalParams=Foods(BestInd,:);
    
    iter=1;
    while ((iter <= ABCOpts.MaxCycles)),
        
        %%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:(FoodNumber)
            
            %/*The parameter to be changed is determined randomly*/
            Param2Change=fix(rand*ABCOpts.Dim)+1;
            
            %/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
            neighbour=fix(rand*(FoodNumber))+1;
            
            %/*Randomly selected solution must be different from the solution i*/
            while(neighbour==i)
                neighbour=fix(rand*(FoodNumber))+1;
            end;
            
            sol=Foods(i,:);
            %  /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
            sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
            
            %  /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
            ind=find(sol<ABCOpts.lb);
            sol(ind)=ABCOpts.lb(ind);
            ind=find(sol>ABCOpts.ub);
            sol(ind)=ABCOpts.ub(ind);
            
            %evaluate new solution
            
            cmd = ['-v ',num2str(ABCOpts.v),' -c ',num2str(sol(1)),' -g ',num2str(sol(2)),' -p ',num2str(sol(3)),' -s 3'];
            ObjValSol = svmtrain(train_label,train_data,cmd);
            FitnessSol=calculateFitness(ObjValSol);
            
            
            %         ObjValSol=feval(objfun,sol);
            %         FitnessSol=calculateFitness(ObjValSol);
            
            % /*a greedy selection is applied between the current solution i and its mutant*/
            if (FitnessSol>Fitness(i)) %/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                Foods(i,:)=sol;
                Fitness(i)=FitnessSol;
                ObjVal(i)=ObjValSol;
                trial(i)=0;
            else
                trial(i)=trial(i)+1; %/*if the solution i can not be improved, increase its trial counter*/
            end;
            
            
        end;
        
        %%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %/* A food source is chosen with the probability which is proportioal to its quality*/
        %/*Different schemes can be used to calculate the probability values*/
        %/*For example prob(i)=fitness(i)/sum(fitness)*/
        %/*or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b*/
        %/*probability values are calculated by using fitness values and normalized by dividing maximum fitness value*/
        
        prob=(0.9.*Fitness./max(Fitness))+0.1;
        
        %%%%%%%%%%%%%%%%%%%%%%%% ONLOOKER BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        i=1;
        t=0;
        while(t<FoodNumber)
            if(rand<prob(i))
                t=t+1;
                %/*The parameter to be changed is determined randomly*/
                Param2Change=fix(rand*ABCOpts.Dim)+1;
                
                %/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
                neighbour=fix(rand*(FoodNumber))+1;
                
                %/*Randomly selected solution must be different from the solution i*/
                while(neighbour==i)
                    neighbour=fix(rand*(FoodNumber))+1;
                end;
                
                sol=Foods(i,:);
                %  /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
                sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
                
                %  /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
                ind=find(sol<ABCOpts.lb);
                sol(ind)=ABCOpts.lb(ind);
                ind=find(sol>ABCOpts.ub);
                sol(ind)=ABCOpts.ub(ind);
                
                %evaluate new solution
                cmd = ['-v ',num2str(ABCOpts.v),' -c ',num2str(sol(1)),' -g ',num2str(sol(2)),' -p ',num2str(sol(3)),' -s 3'];
                ObjValSol = svmtrain(train_label,train_data,cmd);
                FitnessSol=calculateFitness(ObjValSol);
                %         ObjValSol=feval(objfun,sol);
                %         FitnessSol=calculateFitness(ObjValSol);
                
                % /*a greedy selection is applied between the current solution i and its mutant*/
                if (FitnessSol>Fitness(i)) %/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                    Foods(i,:)=sol;
                    Fitness(i)=FitnessSol;
                    ObjVal(i)=ObjValSol;
                    trial(i)=0;
                else
                    trial(i)=trial(i)+1; %/*if the solution i can not be improved, increase its trial counter*/
                end;
            end;
            
            i=i+1;
            if (i==(FoodNumber)+1)
                i=1;
            end;
        end;
        
        
        %/*The best food source is memorized*/
        ind=find(ObjVal==min(ObjVal));
        ind=ind(end);
        if (ObjVal(ind)<GlobalMin)
            GlobalMin=ObjVal(ind);
            GlobalParams=Foods(ind,:);
        end;
        
        
        %%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %/*determine the food sources whose trial counter exceeds the "limit" value.
        %In Basic ABC, only one scout is allowed to occur in each cycle*/
        
        ind=find(trial==max(trial));
        ind=ind(end);
        if (trial(ind)>ABCOpts.Limit)
            Bas(ind)=0;
            sol=(ABCOpts.ub-ABCOpts.lb).*rand(1,ABCOpts.Dim)+ABCOpts.lb;
            cmd = ['-v ',num2str(ABCOpts.v),' -c ',num2str(sol(1)),' -g ',num2str(sol(2)),' -p ',num2str(sol(3)),' -s 3'];
            ObjValSol = svmtrain(train_label,train_data,cmd);
            FitnessSol=calculateFitness(ObjValSol);
            %     ObjValSol=feval(objfun,sol);
            %     FitnessSol=calculateFitness(ObjValSol);
            Foods(ind,:)=sol;
            Fitness(ind)=FitnessSol;
            ObjVal(ind)=ObjValSol;
        end;
        
        trace(iter,1) = GlobalMin;
        trace(iter,2) = sum(ObjVal)/length(ObjVal);
        
        fprintf('iter=%d Params= %4.4f,%4.4f,%4.4f ObjVal=%4.4f\n',iter,GlobalParams,GlobalMin);
        iter=iter+1;
        
    end % End of ABC
    
    GlobalMins(r)=GlobalMin;
end; %end of runs
BestParams=GlobalParams;
BestMSE=GlobalMin;

figure;
hold on;
trace = round(trace*10000)/10000;
plot(trace(1:iter-1,1),'r*-','LineWidth',1.5);
plot(trace(1:iter-1,2),'o-','LineWidth',1.5);
legend('最佳适应度','平均适应度');
xlabel('代数','FontSize',12);
ylabel('适应度','FontSize',12);
grid on;
axis auto;

line1 = '适应度曲线MSE[ABCmethod]';
line2 = ['终止代数=', ...
    num2str(iter-1),',种群数量NP=', ...
    num2str(ABCOpts.ColonySize),')'];
line3 = ['Best c=',num2str(GlobalParams(1)),' g=',num2str(GlobalParams(2)),' p=',num2str(GlobalParams(3)), ...
    ' MSE=',num2str(GlobalMin)];
title({line1;line2;line3},'FontSize',12);
save all
