% Á£×ÓÈºËã·¨
% pso_Trelea_vectorized.m
% a generic particle swarm optimizer
% to find the minimum or maximum of any 
% MISO matlab function
%
% Implements Common, Trelea type 1 and 2, and Clerc's class 1". It will
% also automatically try to track to a changing environment (with varied
% success - BKB 3/18/05)
%
% This vectorized version removes the for loop associated with particle
% number. It also *requires* that the cost function have a single input
% that represents all dimensions of search (i.e., for a function that has 2
% inputs then make a wrapper that passes a matrix of ps x 2 as a single
% variable)
%
% Usage:
%  [optOUT]=PSO(functname,D)
% or:
%  [optOUT,tr,te]=...
%        PSO(functname,D,mv,VarRange,minmax,PSOparams,plotfcn,PSOseedValue)
%
% Inputs:
%    functname - string of matlab function to optimize
%    D - # of inputs to the function (dimension of problem)
%    
% Optional Inputs:
%    mv - max particle velocity, either a scalar or a vector of length D
%           (this allows each component to have it's own max velocity), 
%           default = 4, set if not input or input as NaN
%
%    VarRange - matrix of ranges for each input variable, 
%      default -100 to 100, of form:
%       [ min1 max1 
%         min2 max2
%            ...
%         minD maxD ]
%
%    minmax = 0, funct minimized (default)
%           = 1, funct maximized
%           = 2, funct is targeted to P(12) (minimizes distance to errgoal)
%
%    PSOparams - PSO parameters
%      P(1) - Epochs between updating display, default = 100. if 0, 
%             no display
%      P(2) - Maximum number of iterations (epochs) to train, default = 2000.
%      P(3) - population size, default = 24
%
%      P(4) - acceleration const 1 (local best influence), default = 2
%      P(5) - acceleration const 2 (global best influence), default = 2
%      P(6) - Initial inertia weight, default = 0.9
%      P(7) - Final inertia weight, default = 0.4
%      P(8) - Epoch when inertial weight at final value, default = 1500
%      P(9)- minimum global error gradient, 
%                 if abs(Gbest(i+1)-Gbest(i)) < gradient over 
%                 certain length of epochs, terminate run, default = 1e-25
%      P(10)- epochs before error gradient criterion terminates run, 
%                 default = 150, if the SSE does not change over 250 epochs
%                               then exit
%      P(11)- error goal, if NaN then unconstrained min or max, default=NaN
%      P(12)- type flag (which kind of PSO to use)
%                 0 = Common PSO w/intertia (default)
%                 1,2 = Trelea types 1,2
%                 3   = Clerc's Constricted PSO, Type 1"
%      P(13)- PSOseed, default=0
%               = 0 for initial positions all random
%               = 1 for initial particles as user input
%
%    plotfcn - optional name of plotting function, default 'goplotpso',
%              make your own and put here
%
%    PSOseedValue - initial particle position, depends on P(13), must be
%                   set if P(13) is 1 or 2, not used for P(13)=0, needs to
%                   be nXm where n<=ps, and m<=D
%                   If n<ps and/or m<D then remaining values are set random
%                   on Varrange
% Outputs:
%    optOUT - optimal inputs and associated min/max output of function, of form:
%        [ bestin1
%          bestin2
%            ...
%          bestinD
%          bestOUT ]
%
% Optional Outputs:
%    tr    - Gbest at every iteration, traces flight of swarm
%    te    - epochs to train, returned as a vector 1:endepoch
%
% Example:  out=pso('Ackley')
 
% Brian Birge
% Rev 3.3
% 2/18/06
 
function [OUT,varargout]=PSO(functname,D,varargin)
 
rand('state',sum(100*clock));
if nargin < 1
   error('Not enough arguments.');
end
 
 
% PSO PARAMETERS
if nargin == 1      % only specified functname and D
    D=2
   VRmin=ones(D,1)*-100; 
   VRmax=ones(D,1)*100;    
   VR=[VRmin,VRmax];
   minmax = 0;
   P = [];
   mv = 4;
   plotfcn='goplotpso';   
elseif nargin == 3  % specified functname, D, and mv
   VRmin=ones(D,1)*-100; 
   VRmax=ones(D,1)*100;    
   VR=[VRmin,VRmax];
   minmax = 0;
   mv=varargin{1};
   if isnan(mv)
       mv=4;
   end
   P = [];
   plotfcn='goplotpso';   
elseif nargin == 4  % specified functname, D, mv, Varrange
   mv=varargin{1};
   if isnan(mv)
       mv=4;
   end
   VR=varargin{2}; 
   minmax = 0;
   P = [];
   plotfcn='goplotpso';   
elseif nargin == 5  % Functname, D, mv, Varrange, and minmax
   mv=varargin{1};
   if isnan(mv)
       mv=4;
   end    
   VR=varargin{2};
   minmax=varargin{3};
   P = [];
   plotfcn='goplotpso';
elseif nargin == 6  % Functname, D, mv, Varrange, minmax, and psoparams
   mv=varargin{1};
   if isnan(mv)
       mv=4;
   end    
   VR=varargin{2};
   minmax=varargin{3};
   P = varargin{4}; % psoparams
   plotfcn='goplotpso';   
elseif nargin == 7  % Functname, D, mv, Varrange, minmax, and psoparams, plotfcn
   mv=varargin{1};
   if isnan(mv)
       mv=4;
   end    
   VR=varargin{2};
   minmax=varargin{3};
   P = varargin{4}; % psoparams
   plotfcn = varargin{5}; 
elseif nargin == 8  % Functname, D, mv, Varrange, minmax, and psoparams, plotfcn, PSOseedValue
   mv=varargin{1};
   if isnan(mv)
       mv=4;
   end    
   VR=varargin{2};
   minmax=varargin{3};
   P = varargin{4}; % psoparams
   plotfcn = varargin{5};  
   PSOseedValue = varargin{6};
else    
   error('Wrong # of input arguments.');
end
 
% sets up default pso params
Pdef = [100 2000 100 2 2 0.9 0.4 1500 1e-25 250 NaN 0 0];
Plen = length(P);
P    = [P,Pdef(Plen+1:end)];
 
df      = P(1);
me      = P(2);
ps      = P(3);
ac1     = P(4);
ac2     = P(5);
iw1     = P(6);
iw2     = P(7);
iwe     = P(8);
ergrd   = P(9);
ergrdep = P(10);
errgoal = P(11);
trelea  = P(12);
PSOseed = P(13);
 
% used with trainpso, for neural net training
if strcmp(functname,'pso_neteval')
   net = evalin('caller','net');
    Pd = evalin('caller','Pd');
    Tl = evalin('caller','Tl');
    Ai = evalin('caller','Ai');
     Q = evalin('caller','Q');
    TS = evalin('caller','TS');
end
 
 
% error checking
 if ((minmax==2) & isnan(errgoal))
     error('minmax= 2, errgoal= NaN: choose an error goal or set minmax to 0 or 1');
 end
 
 if ( (PSOseed==1) & ~exist('PSOseedValue') )
     error('PSOseed flag set but no PSOseedValue was input');
 end
 
 if exist('PSOseedValue')
     tmpsz=size(PSOseedValue);
     if D < tmpsz(2)
         error('PSOseedValue column size must be D or less');
     end
     if ps < tmpsz(1)
         error('PSOseedValue row length must be # of particles or less');
     end
 end
 
% set plotting flag
if (P(1))~=0
  plotflg=1;
else
  plotflg=0;
end
 
% preallocate variables for speed up
 tr = ones(1,me)*NaN;
 
% take care of setting max velocity and position params here
if length(mv)==1
 velmaskmin = -mv*ones(ps,D);     % min vel, psXD matrix
 velmaskmax = mv*ones(ps,D);      % max vel
elseif length(mv)==D     
 velmaskmin = repmat(forcerow(-mv),ps,1); % min vel
 velmaskmax = repmat(forcerow( mv),ps,1); % max vel
else
 error('Max vel must be either a scalar or same length as prob dimension D');
end
posmaskmin  = repmat(VR(1:D,1)',ps,1);  % min pos, psXD matrix
posmaskmax  = repmat(VR(1:D,2)',ps,1);  % max pos
posmaskmeth = 3; % 3=bounce method (see comments below inside epoch loop)
 
% PLOTTING
 message = sprintf('PSO: %%g/%g iterations, GBest = %%20.20g.\n',me);
 
% INITIALIZE INITIALIZE INITIALIZE INITIALIZE INITIALIZE INITIALIZE
 
% initialize population of particles and their velocities at time zero,
% format of pos= (particle#, dimension)
 % construct random population positions bounded by VR
  pos(1:ps,1:D) = normmat(rand([ps,D]),VR',1);
  
  if PSOseed == 1         % initial positions user input, see comments above
    tmpsz                      = size(PSOseedValue);
    pos(1:tmpsz(1),1:tmpsz(2)) = PSOseedValue;  
  end
 
 % construct initial random velocities between -mv,mv
  vel(1:ps,1:D) = normmat(rand([ps,D]),...
      [forcecol(-mv),forcecol(mv)]',1);
 
% initial pbest positions vals
 pbest = pos;
 
% VECTORIZE THIS, or at least vectorize cost funct call 
 out = feval(functname,pos);  % returns column of cost values (1 for each particle)
%---------------------------
 
 pbestval=out;   % initially, pbest is same as pos
 
% assign initial gbest here also (gbest and gbestval)
 if minmax==1
   % this picks gbestval when we want to maximize the function
    [gbestval,idx1] = max(pbestval);
 elseif minmax==0
   % this works for straight minimization
    [gbestval,idx1] = min(pbestval);
 elseif minmax==2
   % this works when you know target but not direction you need to go
   % good for a cost function that returns distance to target that can be either
   % negative or positive (direction info)
    [temp,idx1] = min((pbestval-ones(size(pbestval))*errgoal).^2);
    gbestval    = pbestval(idx1);
 end
 
 % preallocate a variable to keep track of gbest for all iters
 bestpos        = zeros(me,D+1)*NaN;
 gbest          = pbest(idx1,:);  % this is gbest position
   % used with trainpso, for neural net training
   % assign gbest to net at each iteration, these interim assignments
   % are for plotting mostly
    if strcmp(functname,'pso_neteval')
        net=setx(net,gbest);
    end
 %tr(1)          = gbestval;       % save for output
 bestpos(1,1:D) = gbest;
 
% this part used for implementing Carlisle and Dozier's APSO idea
% slightly modified, this tracks the global best as the sentry whereas
% their's chooses a different point to act as sentry
% see "Tracking Changing Extremea with Adaptive Particle Swarm Optimizer",
% part of the WAC 2002 Proceedings, June 9-13, http://wacong.com
 sentryval = gbestval;
 sentry    = gbest;
 
if (trelea == 3)
% calculate Clerc's constriction coefficient chi to use in his form
 kappa   = 1; % standard val = 1, change for more or less constriction    
 if ( (ac1+ac2) <=4 )
     chi = kappa;
 else
     psi     = ac1 + ac2;
     chi_den = abs(2-psi-sqrt(psi^2 - 4*psi));
     chi_num = 2*kappa;
     chi     = chi_num/chi_den;
 end
end
 
% INITIALIZE END INITIALIZE END INITIALIZE END INITIALIZE END
rstflg = 0; % for dynamic environment checking
% start PSO iterative procedures
 cnt    = 0; % counter used for updating display according to df in the options
 cnt2   = 0; % counter used for the stopping subroutine based on error convergence
 iwt(1) = iw1;
for i=1:me  % start epoch loop (iterations)
 
     out        = feval(functname,[pos;gbest]);
     outbestval = out(end,:);
     out        = out(1:end-1,:);
 
     tr(i+1)          = gbestval; % keep track of global best val
     te               = i; % returns epoch number to calling program when done
     bestpos(i,1:D+1) = [gbest,gbestval];
     
     %assignin('base','bestpos',bestpos(i,1:D+1));
   %------------------------------------------------------------------------      
   % this section does the plots during iterations   
    if plotflg==1      
      if (rem(i,df) == 0 ) | (i==me) | (i==1) 
         fprintf(message,i,gbestval);
         cnt = cnt+1; % count how many times we display (useful for movies)
          
         eval(plotfcn); % defined at top of script
         
      end  % end update display every df if statement    
    end % end plotflg if statement
 
    % check for an error space that changes wrt time/iter
    % threshold value that determines dynamic environment 
    % sees if the value of gbest changes more than some threshold value
    % for the same location
    chkdyn = 1;
    rstflg = 0; % for dynamic environment checking
 
    if chkdyn==1
     threshld = 0.05;  % percent current best is allowed to change, .05 = 5% etc
     letiter  = 5; % # of iterations before checking environment, leave at least 3 so PSO has time to converge
     outorng  = abs( 1- (outbestval/gbestval) ) >= threshld;
     samepos  = (max( sentry == gbest ));
 
     if (outorng & samepos) & rem(i,letiter)==0
         rstflg=1;
       % disp('New Environment: reset pbest, gbest, and vel');
       %% reset pbest and pbestval if warranted
%        outpbestval = feval( functname,[pbest] );
%        Poutorng    = abs( 1-(outpbestval./pbestval) ) > threshld;
%        pbestval    = pbestval.*~Poutorng + outpbestval.*Poutorng;
%        pbest       = pbest.*repmat(~Poutorng,1,D) + pos.*repmat(Poutorng,1,D);   
 
        pbest     = pos; % reset personal bests to current positions
        pbestval  = out; 
        vel       = vel*10; % agitate particles a little (or a lot)
        
       % recalculate best vals 
        if minmax == 1
           [gbestval,idx1] = max(pbestval);
        elseif minmax==0
           [gbestval,idx1] = min(pbestval);
        elseif minmax==2 % this section needs work
           [temp,idx1] = min((pbestval-ones(size(pbestval))*errgoal).^2);
           gbestval    = pbestval(idx1);
        end
        
        gbest  = pbest(idx1,:);
        
        % used with trainpso, for neural net training
        % assign gbest to net at each iteration, these interim assignments
        % are for plotting mostly
        if strcmp(functname,'pso_neteval')
           net=setx(net,gbest);
        end
     end  % end if outorng
     
     sentryval = gbestval;
     sentry    = gbest;
     
    end % end if chkdyn
    
    % find particles where we have new pbest, depending on minmax choice 
    % then find gbest and gbestval
     %[size(out),size(pbestval)]
    if rstflg == 0
     if minmax == 0
        [tempi]            = find(pbestval>=out); % new min pbestvals
        pbestval(tempi,1)  = out(tempi);   % update pbestvals
        pbest(tempi,:)     = pos(tempi,:); % update pbest positions
       
        [iterbestval,idx1] = min(pbestval);
        
        if gbestval >= iterbestval
            gbestval = iterbestval;
            gbest    = pbest(idx1,:);
            % used with trainpso, for neural net training
            % assign gbest to net at each iteration, these interim assignments
            % are for plotting mostly
             if strcmp(functname,'pso_neteval')
                net=setx(net,gbest);
             end
        end
     elseif minmax == 1
        [tempi,dum]        = find(pbestval<=out); % new max pbestvals
        pbestval(tempi,1)  = out(tempi,1); % update pbestvals
        pbest(tempi,:)     = pos(tempi,:); % update pbest positions
 
        [iterbestval,idx1] = max(pbestval);
        if gbestval <= iterbestval
            gbestval = iterbestval;
            gbest    = pbest(idx1,:);
            % used with trainpso, for neural net training
            % assign gbest to net at each iteration, these interim assignments
            % are for plotting mostly
             if strcmp(functname,'pso_neteval')
                net=setx(net,gbest);
             end
        end
     elseif minmax == 2  % this won't work as it is, fix it later
        egones            = errgoal*ones(ps,1); % vector of errgoals
        sqrerr2           = ((pbestval-egones).^2);
        sqrerr1           = ((out-egones).^2);
        [tempi,dum]       = find(sqerr1 <= sqrerr2); % find particles closest to targ
        pbestval(tempi,1) = out(tempi,1); % update pbestvals
        pbest(tempi,:)    = pos(tempi,:); % update pbest positions
 
        sqrerr            = ((pbestval-egones).^2); % need to do this to reflect new pbests
        [temp,idx1]       = min(sqrerr);
        iterbestval       = pbestval(idx1);
        
        if (iterbestval-errgoal)^2 <= (gbestval-errgoal)^2
           gbestval = iterbestval;
           gbest    = pbest(idx1,:);
           % used with trainpso, for neural net training
            % assign gbest to net at each iteration, these interim assignments
            % are for plotting mostly
             if strcmp(functname,'pso_neteval')
                net=setx(net,gbest);
             end
        end
     end
    end
    
    
 %   % build a simple predictor 10th order, for gbest trajectory
 %   if i>500
 %    for dimcnt=1:D
 %      pred_coef  = polyfit(i-250:i,(bestpos(i-250:i,dimcnt))',20);
 %     % pred_coef  = polyfit(200:i,(bestpos(200:i,dimcnt))',20);       
 %      gbest_pred(i,dimcnt) = polyval(pred_coef,i+1);
 %    end
 %    else 
%       gbest_pred(i,:) = zeros(size(gbest));
%    end
  
   %gbest_pred(i,:)=gbest;    
   %assignin('base','gbest_pred',gbest_pred);
 
 %   % convert to non-inertial frame
 %    gbestoffset = gbest - gbest_pred(i,:);
 %    gbest = gbest - gbestoffset;
 %    pos   = pos + repmat(gbestoffset,ps,1);
 %    pbest = pbest + repmat(gbestoffset,ps,1);
 
     %PSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSO
 
      % get new velocities, positions (this is the heart of the PSO algorithm)     
      % each epoch get new set of random numbers
       rannum1 = rand([ps,D]); % for Trelea and Clerc types
       rannum2 = rand([ps,D]);       
       if     trelea == 2    
        % from Trelea's paper, parameter set 2
         vel = 0.729.*vel...                              % prev vel
               +1.494.*rannum1.*(pbest-pos)...            % independent
               +1.494.*rannum2.*(repmat(gbest,ps,1)-pos); % social  
       elseif trelea == 1
        % from Trelea's paper, parameter set 1                     
         vel = 0.600.*vel...                              % prev vel
               +1.700.*rannum1.*(pbest-pos)...            % independent
               +1.700.*rannum2.*(repmat(gbest,ps,1)-pos); % social 
       elseif trelea ==3
        % Clerc's Type 1" PSO
         vel = chi*(vel...                                % prev vel
               +ac1.*rannum1.*(pbest-pos)...              % independent
               +ac2.*rannum2.*(repmat(gbest,ps,1)-pos)) ; % social          
       else
        % common PSO algo with inertia wt 
        % get inertia weight, just a linear funct w.r.t. epoch parameter iwe
         if i<=iwe
            iwt(i) = ((iw2-iw1)/(iwe-1))*(i-1)+iw1;
         else
            iwt(i) = iw2;
         end
        % random number including acceleration constants
         ac11 = rannum1.*ac1;    % for common PSO w/inertia
         ac22 = rannum2.*ac2;
         
         vel = iwt(i).*vel...                             % prev vel
               +ac11.*(pbest-pos)...                      % independent
               +ac22.*(repmat(gbest,ps,1)-pos);           % social                  
       end
       
       % limit velocities here using masking
        vel = ( (vel <= velmaskmin).*velmaskmin ) + ( (vel > velmaskmin).*vel );
        vel = ( (vel >= velmaskmax).*velmaskmax ) + ( (vel < velmaskmax).*vel );     
        
       % update new position (PSO algo)    
        pos = pos + vel;
    
       % position masking, limits positions to desired search space
       % method: 0) no position limiting, 1) saturation at limit,
       %         2) wraparound at limit , 3) bounce off limit
        minposmask_throwaway = pos <= posmaskmin;  % these are psXD matrices
        minposmask_keep      = pos >  posmaskmin;     
        maxposmask_throwaway = pos >= posmaskmax;
        maxposmask_keep      = pos <  posmaskmax;
     
        if     posmaskmeth == 1
         % this is the saturation method
          pos = ( minposmask_throwaway.*posmaskmin ) + ( minposmask_keep.*pos );
          pos = ( maxposmask_throwaway.*posmaskmax ) + ( maxposmask_keep.*pos );      
        elseif posmaskmeth == 2
         % this is the wraparound method
          pos = ( minposmask_throwaway.*posmaskmax ) + ( minposmask_keep.*pos );
          pos = ( maxposmask_throwaway.*posmaskmin ) + ( maxposmask_keep.*pos );                
        elseif posmaskmeth == 3
         % this is the bounce method, particles bounce off the boundaries with -vel      
          pos = ( minposmask_throwaway.*posmaskmin ) + ( minposmask_keep.*pos );
          pos = ( maxposmask_throwaway.*posmaskmax ) + ( maxposmask_keep.*pos );
 
          vel = (vel.*minposmask_keep) + (-vel.*minposmask_throwaway);
          vel = (vel.*maxposmask_keep) + (-vel.*maxposmask_throwaway);
        else
         % no change, this is the original Eberhart, Kennedy method, 
         % it lets the particles grow beyond bounds if psoparams (P)
         % especially Vmax, aren't set correctly, see the literature
        end
 
     %PSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSOPSO
% check for stopping criterion based on speed of convergence to desired 
   % error   
    tmp1 = abs(tr(i) - gbestval);
    if tmp1 > ergrd
       cnt2 = 0;
    elseif tmp1 <= ergrd
       cnt2 = cnt2+1;
       if cnt2 >= ergrdep 
         if plotflg == 1
          fprintf(message,i,gbestval);           
          disp(' ');
          disp(['--> Solution likely, GBest hasn''t changed by at least ',...
              num2str(ergrd),' for ',...
                  num2str(cnt2),' epochs.']);  
          eval(plotfcn);
         end       
         break
       end
    end
    
   % this stops if using constrained optimization and goal is reached
    if ~isnan(errgoal)
     if ((gbestval<=errgoal) & (minmax==0)) | ((gbestval>=errgoal) & (minmax==1))  
 
         if plotflg == 1
             fprintf(message,i,gbestval);
             disp(' ');            
             disp(['--> Error Goal reached, successful termination!']);
             
             eval(plotfcn);
         end
         break
     end
     
    % this is stopping criterion for constrained from both sides    
     if minmax == 2
       if ((tr(i)<errgoal) & (gbestval>=errgoal)) | ((tr(i)>errgoal) ...
               & (gbestval <= errgoal))        
         if plotflg == 1
             fprintf(message,i,gbestval);
             disp(' ');            
             disp(['--> Error Goal reached, successful termination!']);            
             
             eval(plotfcn);
         end
         break              
       end
     end % end if minmax==2
    end  % end ~isnan if
 
 %    % convert back to inertial frame
 %     pos = pos - repmat(gbestoffset,ps,1);
 %     pbest = pbest - repmat(gbestoffset,ps,1);
 %     gbest = gbest + gbestoffset;
  
 
end  % end epoch loop
 
%% clear temp outputs
% evalin('base','clear temp_pso_out temp_te temp_tr;');
 
% output & return
 OUT=[gbest';gbestval];
 varargout{1}=[1:te];
 varargout{2}=[tr(find(~isnan(tr)))];
 
 return

% forcecol.m
% function to force a vector to be a single column
%
% Brian Birge
% Rev 1.0
% 7/1/98
 
function[out]=forcecol(in)
len=prod(size(in));
out=reshape(in,[len,1]);


% goplotpso.m
% default plotting script used in PSO functions
%
% this script is not a function,
% it is a plugin for the main PSO routine (pso_Trelea_vectorized)
% so it shares all the same variables, be careful with variable names
% when making your own plugin
 
% Brian Birge
% Rev 2.0
% 3/1/06
 
% setup figure, change this for your own machine
 clf
 set(gcf,'Position',[651    31   626   474]); % this is the computer dependent part
 %set(gcf,'Position',[743    33   853   492]);
 set(gcf,'Doublebuffer','on');
               
% particle plot, upper right
 subplot('position',[.7,.6,.27,.32]);
 set(gcf,'color','k')
 
 plot3(pos(:,1),pos(:,D),out,'b.','Markersize',7)
 
 hold on
 plot3(pbest(:,1),pbest(:,D),pbestval,'g.','Markersize',7);
 plot3(gbest(1),gbest(D),gbestval,'r.','Markersize',25);
 
 % crosshairs
 offx = max(abs(min(min(pbest(:,1)),min(pos(:,1)))),...
            abs(max(max(pbest(:,1)),max(pos(:,1)))));
 
 offy = max(abs(min(min(pbest(:,D)),min(pos(:,D)))),...
            abs(min(max(pbest(:,D)),max(pos(:,D)))));
 plot3([gbest(1)-offx;gbest(1)+offx],...
       [gbest(D);gbest(D)],...
       [gbestval;gbestval],...
       'r-.');
 plot3([gbest(1);gbest(1)],...
       [gbest(D)-offy;gbest(D)+offy],...
       [gbestval;gbestval],...
       'r-.');
    
 hold off
 
 xlabel('Dimension 1','color','y')
 ylabel(['Dimension ',num2str(D)],'color','y')
 zlabel('Cost','color','y')
 
 title('Particle Dynamics','color','w','fontweight','bold')
 
 set(gca,'Xcolor','y')
 set(gca,'Ycolor','y')
 set(gca,'Zcolor','y')
 set(gca,'color','k')
            
 % camera control
 view(2)
 try
   axis([gbest(1)-offx,gbest(1)+offx,gbest(D)-offy,gbest(D)+offy]);
 catch
   axis([VR(1,1),VR(1,2),VR(D,1),VR(D,2)]);
 end
 
% error plot, left side
 subplot('position',[0.1,0.1,.475,.825]);
  semilogy(tr(find(~isnan(tr))),'color','m','linewidth',2)
  %plot(tr(find(~isnan(tr))),'color','m','linewidth',2)
  xlabel('epoch','color','y')
  ylabel('gbest val.','color','y')
  
  if D==1
     titstr1=sprintf(['%11.6g = %s( [ %9.6g ] )'],...
                gbestval,strrep(functname,'_','\_'),gbest(1));
  elseif D==2
     titstr1=sprintf(['%11.6g = %s( [ %9.6g, %9.6g ] )'],...
                gbestval,strrep(functname,'_','\_'),gbest(1),gbest(2));
  elseif D==3
     titstr1=sprintf(['%11.6g = %s( [ %9.6g, %9.6g, %9.6g ] )'],...
                gbestval,strrep(functname,'_','\_'),gbest(1),gbest(2),gbest(3));
  else
     titstr1=sprintf(['%11.6g = %s( [ %g inputs ] )'],...
                gbestval,strrep(functname,'_','\_'),D);
  end
  title(titstr1,'color','m','fontweight','bold');
  
  grid on
%  axis tight
 
  set(gca,'Xcolor','y')
  set(gca,'Ycolor','y')
  set(gca,'Zcolor','y')
  set(gca,'color','k')
 
  set(gca,'YMinorGrid','off')
  
% text box in lower right
% doing it this way so I can format each line any way I want
subplot('position',[.62,.1,.29,.4]);
  clear titstr
  if trelea==0
       PSOtype  = 'Common PSO';
       xtraname = 'Inertia Weight : ';
       xtraval  = num2str(iwt(length(iwt)));
       
     elseif trelea==2 | trelea==1
       
       PSOtype  = (['Trelea Type ',num2str(trelea)]);
       xtraname = ' ';
       xtraval  = ' ';
       
     elseif trelea==3
       PSOtype  = (['Clerc Type 1"']);
       xtraname = '\chi value : ';
       xtraval  = num2str(chi);
 
  end
  if isnan(errgoal)
    errgoalstr='Unconstrained';
  else
    errgoalstr=num2str(errgoal);
  end
  if minmax==1
     minmaxstr = ['Maximize to : '];
  elseif minmax==0
     minmaxstr = ['Minimize to : '];
  else
     minmaxstr = ['Target   to : '];
  end
  
  if rstflg==1
     rststat1 = 'Environment Change';
     rststat2 = ' ';
  else
     rststat1 = ' ';
     rststat2 = ' ';
  end
  
  titstr={'PSO Model: '      ,PSOtype;...
          'Dimensions : '    ,num2str(D);...
          '# of particles : ',num2str(ps);...
          minmaxstr          ,errgoalstr;...
          'Function : '      ,strrep(functname,'_','\_');...
          xtraname           ,xtraval;...
          rststat1           ,rststat2};
  
  text(.1,1,[titstr{1,1},titstr{1,2}],'color','g','fontweight','bold');
  hold on
  text(.1,.9,[titstr{2,1},titstr{2,2}],'color','m');
  text(.1,.8,[titstr{3,1},titstr{3,2}],'color','m');
  text(.1,.7,[titstr{4,1}],'color','w');
  text(.55,.7,[titstr{4,2}],'color','m');
  text(.1,.6,[titstr{5,1},titstr{5,2}],'color','m');
  text(.1,.5,[titstr{6,1},titstr{6,2}],'color','w','fontweight','bold');
  text(.1,.4,[titstr{7,1},titstr{7,2}],'color','r','fontweight','bold');
  
  % if we are training a neural net, show a few more parameters
  if strcmp('pso_neteval',functname)
    % net is passed from trainpso to pso_Trelea_vectorized in case you are
    % wondering where that structure comes from
    hiddlyrstr = [];  
    for lyrcnt=1:length(net.layers)
       TF{lyrcnt} = net.layers{lyrcnt}.transferFcn;
       Sn(lyrcnt) = net.layers{lyrcnt}.dimensions;
       hiddlyrstr = [hiddlyrstr,', ',TF{lyrcnt}];
    end
    hiddlyrstr = hiddlyrstr(3:end);
  
    text(0.1,.35,['#neur/lyr = [ ',num2str(net.inputs{1}.size),'  ',...
               num2str(Sn),' ]'],'color','c','fontweight','normal',...
               'fontsize',10);   
    text(0.1,.275,['Lyr Fcn: ',hiddlyrstr],...
       'color','c','fontweight','normal','fontsize',9);
       
  end
  
  
  legstr = {'Green = Personal Bests';...
            'Blue  = Current Positions';...
            'Red   = Global Best'};
  text(.1,0.025,legstr{1},'color','g');
  text(.1,-.05,legstr{2},'color','b');
  text(.1,-.125,legstr{3},'color','r');
  
  hold off
 
  set(gca,'color','k');
  set(gca,'visible','off');
  
  drawnow


function [out,varargout]=normmat(x,newminmax,flag)
% normmat.m
% takes a matrix and reformats the data to fit between a new range
% 
% Usage:
%    [xprime,mins,maxs]=normmat(x,range,method)
%
% Inputs:
%     x - matrix to reformat of dimension MxN
%     range - a vector or matrix specifying minimum and maximum values for the new matrix
%         for method = 0, range is a 2 element row vector of [min,max]
%         for method = 1, range is a 2 row matrix with same column size as 
%                         input matrix with format of [min1,min2,...minN;
%                                                      max1,max2,...maxM];
%         for method = 2, range is a 2 column matrix with same row size as
%                         input matrix with format of [min1,max1;
%                                                      min2,max2;
%                                                      ... , ...;
%                                                      minM,maxM];
%             alternatively for method 1 and 2, can input just a 2 element vector as in method 0
%             this will just apply the same min/max across each column or row respectively
%     method - a scalar flag with the following function
%         = 1, normalize each column of the input matrix separately
%         = 2, normalize each row of the input matrix separately
%         = 0, normalize matrix globally
% Outputs:
%     xprime - new matrix normalized per method
%     mins,maxs - optional outputs return the min and max vectors of the original matrix x
%         used for recovering original matrix from xprime
%
% example: x = [-10,3,0;2,4.1,-7;3.4,1,0.01]
%          [xprime,mins,maxs]=normmat(x,[0,10],0)
 
% Brian Birge
% Rev 2.1
% 3/16/06 - changed name of function to avoid same name in robust control
% toolbox
%--------------------------------------------------------------------------------------------------------
if flag==0
 
  a=min(min((x)));
  b=max(max((x)));
  if abs(a)>abs(b)
     large=a;
     small=b;
  else
     large=b;
     small=a;
  end
  temp=size(newminmax);
  if temp(1)~=1
     error('Error: for method=0, range vector must be a 2 element row vector');
  end  
  den=abs(large-small);  
  range=newminmax(2)-newminmax(1);
  if den==0
     out=x;
  else     
     z21=(x-a)/(den);  
     out=z21*range+newminmax(1)*ones(size(z21));
  end
  
%--------------------------------------------------------------------------------------------------------  
elseif flag==1
 a=min(x,[],1);
 b=max(x,[],1);
  for i=1:length(b)
     if abs(a(i))>abs(b(i))
        large(i)=a(i);
        small(i)=b(i);
     else
        large(i)=b(i);
        small(i)=a(i);
     end
  end
  den=abs(large-small);
  temp=size(newminmax);
  if temp(1)*temp(2)==2
     newminmaxA(1,:)=newminmax(1).*ones(size(x(1,:)));
     newminmaxA(2,:)=newminmax(2).*ones(size(x(1,:)));
  elseif temp(1)>2
     error('Error: for method=1, range matrix must have 2 rows and same columns as input matrix');
  else
     newminmaxA=newminmax;
  end
  
  range=newminmaxA(2,:)-newminmaxA(1,:);  
  for j=1:length(x(:,1))    
     for i=1:length(b)
        if den(i)==0
           out(j,i)=x(j,i);
        else
           z21(j,i)=(x(j,i)-a(i))./(den(i));
           out(j,i)=z21(j,i).*range(1,i)+newminmaxA(1,i);
        end
     end     
  end  
%--------------------------------------------------------------------------------------------------------  
elseif flag==2
  a=min(x,[],2);
  b=max(x,[],2);
  for i=1:length(b)
     if abs(a(i))>abs(b(i))
        large(i)=a(i);
        small(i)=b(i);
     else
        large(i)=b(i);
        small(i)=a(i);
     end 
  end
  den=abs(large-small);
  temp=size(newminmax);
  if temp(1)*temp(2)==2
     newminmaxA(:,1)=newminmax(1).*ones(size(x(:,1)));
     newminmaxA(:,2)=newminmax(2).*ones(size(x(:,1)));    
  elseif temp(2)>2
     error('Error: for method=2, range matrix must have 2 columns and same rows as input matrix');
  else
     newminmaxA=newminmax;
  end
  
  range=newminmaxA(:,2)-newminmaxA(:,1);  
  for j=1:length(x(1,:))
     for i=1:length(b)
        if den(i)==0
           out(i,j)=x(i,j);
        else           
           z21(i,j)=(x(i,j)-a(i))./([forcecol(den(i))]);
           out(i,j)=z21(i,j).*range(i,1)+newminmaxA(i,1);
        end
     end     
  end  
  
end
%--------------------------------------------------------------------------------------------------------
varargout{1}=a;
varargout{2}=b;
 
return
