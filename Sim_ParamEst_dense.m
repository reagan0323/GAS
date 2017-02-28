% This file applies GAS to estimate parameters in the simulated data
%
% Need to call 
%    Sim_Setting
%
% 12/15/2016 by Gen Li


% for choosesetting=1:4;

choosesetting=1;
sparsity=0;
Nsim=100;
Sim_Setting


%% load simulation settings

close all
rec_Vangle=zeros(Nsim,6); % V0, A1, A2 (GAS, GAS_iter) 
rec_Theta=zeros(Nsim,4); % Theta1 Theta2
rec_signal=zeros(Nsim,12);% Mean1,Mean2, Joint1,Ind1, Joint2,Ind2
rec_time=zeros(Nsim,2); % GAS GAS_iter
num_fail=0;




%% start simulation
nsim=1;
while nsim<=Nsim
    disp(['Simulation run: ',num2str(nsim)]);
    
    
    
    % simulate two data matrices
    rng('shuffle');
    switch distr1
        case 'binomial'
            X1=binornd(1,param1);
        case 'poisson'
            X1=poissrnd(param1);
        case 'normal'
            X1=normrnd(param1,1);
    end;
    switch distr2
        case 'binomial'
            X2=binornd(1,param2);
        case 'poisson'
            X2=poissrnd(param2);
        case 'normal'
            X2=normrnd(param2,1);
    end;


    
    % GAS (one-step procedure, no ridge parameter)
    try 
        tGAS=tic;
        [U0_GAS,U1_GAS,U2_GAS,V0_GAS,A1_GAS,A2_GAS,Mu0_GAS]=...
            GAS(X1,X2,r0,r1,r2,distr1,distr2);  
        Mu1_GAS=Mu0_GAS(1:p1);
        Mu2_GAS=Mu0_GAS((p1+1):end);
        Mean1_GAS=ones(n,1)*Mu1_GAS';
        Mean2_GAS=ones(n,1)*Mu2_GAS';
        V1_GAS=V0_GAS(1:p1,:);
        V2_GAS=V0_GAS((p1+1):end,:);
        Jnt1_GAS=U0_GAS*V1_GAS';
        Jnt2_GAS=U0_GAS*V2_GAS';
        Ind1_GAS=U1_GAS*A1_GAS';
        Ind2_GAS=U2_GAS*A2_GAS';
        Theta1_GAS=Mean1_GAS+Jnt1_GAS+Ind1_GAS;
        Theta2_GAS=Mean2_GAS+Jnt2_GAS+Ind2_GAS;
        T3=toc(tGAS);
    catch
        num_fail=num_fail+1;
        disp('*******************************************************')  ;  
        disp('WARNING: THE DEFAULT GAS FAILS! RESTART SIMULATION.');
        disp('*******************************************************')  ; 
        continue;
    end;

    
    
    
    % GAS_iter (double iterative procedure, need ridge penalty for binomial data)
    try
        tiGAS=tic;
        if strcmp(distr1,'binomial') && ~strcmp(distr2,'binomial')
            [U0_iGAS,U1_iGAS,U2_iGAS,V0_iGAS,A1_iGAS,A2_iGAS,Mu0_iGAS]=...
                GAS(X1,X2,r0,r1,r2,distr1,distr2,struct('inner_Niter',5,'lambda1',1E-3));  
        elseif ~strcmp(distr1,'binomial') && strcmp(distr2,'binomial')
            [U0_iGAS,U1_iGAS,U2_iGAS,V0_iGAS,A1_iGAS,A2_iGAS,Mu0_iGAS]=...
                GAS(X1,X2,r0,r1,r2,distr1,distr2,struct('inner_Niter',5,'lambda2',1E-3));      
        elseif strcmp(distr1,'binomial') && strcmp(distr2,'binomial')
            [U0_iGAS,U1_iGAS,U2_iGAS,V0_iGAS,A1_iGAS,A2_iGAS,Mu0_iGAS]=...
                GAS(X1,X2,r0,r1,r2,distr1,distr2,struct('inner_Niter',5,'lambda1',1E-3,'lambda2',1E-3));
        else
            [U0_iGAS,U1_iGAS,U2_iGAS,V0_iGAS,A1_iGAS,A2_iGAS,Mu0_iGAS]=...
                GAS(X1,X2,r0,r1,r2,distr1,distr2,struct('inner_Niter',5));
        end;
        Mu1_iGAS=Mu0_iGAS(1:p1);
        Mu2_iGAS=Mu0_iGAS((p1+1):end);
        Mean1_iGAS=ones(n,1)*Mu1_iGAS';
        Mean2_iGAS=ones(n,1)*Mu2_iGAS';
        V1_iGAS=V0_iGAS(1:p1,:);
        V2_iGAS=V0_iGAS((p1+1):end,:);
        Jnt1_iGAS=U0_iGAS*V1_iGAS';
        Jnt2_iGAS=U0_iGAS*V2_iGAS';
        Ind1_iGAS=U1_iGAS*A1_iGAS';
        Ind2_iGAS=U2_iGAS*A2_iGAS';
        Theta1_iGAS=Mean1_iGAS+Jnt1_iGAS+Ind1_iGAS;
        Theta2_iGAS=Mean2_iGAS+Jnt2_iGAS+Ind2_iGAS;
        T4=toc(tiGAS);
    catch 
        num_fail=num_fail+1;
        disp('*******************************************************')  ;  
        disp('WARNING: THE DEFAULT GAS-iter FAILS! RESTART SIMULATION.');
        disp('*******************************************************')  ; 
        continue;
    end;
    
 
    
    
         
    % compare GAS with adhoc(=EPCA*2+JIVE)
    rec_Vangle(nsim,:)=[PrinAngle(V0true,V0_GAS),PrinAngle(V0true,V0_iGAS),...
        PrinAngle(A1true,A1_GAS),PrinAngle(A1true,A1_iGAS),...
        PrinAngle(A2true,A2_GAS),PrinAngle(A2true,A2_iGAS)]; 
    rec_Theta(nsim,:)=[norm(Theta1true-Theta1_GAS,'fro'),norm(Theta1true-Theta1_iGAS,'fro'),...
        norm(Theta2true-Theta2_GAS,'fro'),norm(Theta2true-Theta2_iGAS,'fro')]; 
    rec_signal(nsim,:)=[norm(Mean1true-Mean1_GAS,'fro'),norm(Mean1true-Mean1_iGAS,'fro'),...
        norm(Mean2true-Mean2_GAS,'fro'),norm(Mean2true-Mean2_iGAS,'fro'),...    
        norm(Jnt1true-Jnt1_GAS,'fro'),norm(Jnt1true-Jnt1_iGAS,'fro'),...
        norm(Jnt2true-Jnt2_GAS,'fro'),norm(Jnt2true-Jnt2_iGAS,'fro'),...
        norm(Ind1true-Ind1_GAS,'fro'),norm(Ind1true-Ind1_iGAS,'fro'),...
        norm(Ind2true-Ind2_GAS,'fro'),norm(Ind2true-Ind2_iGAS,'fro')];
    rec_time(nsim,:)=[T3,T4]; % EPCA, JIVE, GAS
   
    nsim=nsim+1;
end;
disp([num2str(num_fail),' failures in total to get ',num2str(Nsim),' simulation runs.'])

% comparison
median(rec_Vangle,1);
median(rec_Theta,1);
median(rec_signal,1);
median(rec_time,1);