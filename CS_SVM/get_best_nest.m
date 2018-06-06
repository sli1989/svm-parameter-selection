
%% Find the current best nest
function [fmin,best,nest,fitness]=get_best_nest(nest,newnest,fitness,Ub,Lb)
% Evaluating all new solutions
for j=1:size(nest,1),
    n=25;
   for i=1:n
    newnest(i,find(newnest(i,:)>Ub(1)))=Ub(1);
    newnest(i,find(newnest(i,:)<Lb(1)))=Lb(1);
end
    fnew=fun(newnest(j,:));
  
    if fnew<=fitness(j),
       fitness(j)=fnew;
       
       nest(j,:)=newnest(j,:);
       
    end
end
% Find the current best
[fmin,K]=min(fitness) ;
best=nest(K,:);


