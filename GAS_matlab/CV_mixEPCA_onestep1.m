function [ComCVscore,IndCVscore,total_mis]=CV_mixEPCA_onestep1(X1,X2,distr1,distr2,rcand,MisInd1,MisInd2,paramstruct)
% This function calc CV scores for a mixed-type EPCA problem with two data
% sets (horizontally concatenated)  for a set of ranks  in one realization 
% of multiple missing observations (indicated by MisInd). 
% suitable to est the *total* rank for GCCA (not joint rank)
%
% Note: two major difficulties compared to CV_EPCA
%       1. mixed-type GLM with missing values (no existing solver, need to code our own IRLS)
%       2. combine CV scores from different data types (need to use some weighted avg)
%
% Input
%     X1/X2         n*p1/n*p2 fully observed data matrix, from exponential family
%     distr1/distr2     string, specifying distribution 'normal','binomial','poisson'
%     rcand     a vector of candidate ranks for natural parameter matrix
%     MisInd1/MisInd2    n*p1/n*p2 0/1 matrix, corresponding to X1/X2, 1=missing
%
% Output
%     ComCVscore       a vector of combined CV scores corresp to rcand
%     IndCVscore       2*length(rcand) matrix, first row corresp to CV
%                      score in X1 (avg SS of diff between Xi and Mu_i)
%     total_mis     length-2 vector, total number of missing entries in X1 and X2
%
%
% need to call:
%    
%
% by Gen Li, 11/8/2016


[n,p1]=size(X1);
[n_,p1_]=size(MisInd1);
[n2,p2]=size(X2);
[n2_,p2_]=size(MisInd2);
tX1=X1';
tX2=X2';
tConMisInd=[MisInd1,MisInd2]';
if n~=n_ || n~=n2 || n~=n2_ || p1~=p1_ || p2~=p2_
    error('Missing Index matrix and Data matrix not compatible!');
end;
ComCVscore=zeros(size(rcand));
IndCVscore=zeros(2,length(rcand));
total_mis=[sum(sum(MisInd1)),sum(sum(MisInd2))];
disp(['(Leave out ',num2str(100*total_mis(1)/(n*p1)),'%',...
    ' in X1 and ',num2str(100*total_mis(2)/(n*p2)),'%',...
    ' in X2...)']);

lambda1=1E-3; % default
lambda2=1E-3; % default
Niter=100; % default
Tol=0.1; % default threshold for prinangle of V0
if nargin > 7 % other values
    if isfield(paramstruct,'Niter') ;    
        Niter = getfield(paramstruct,'Niter') ; 
    end ;
    if isfield(paramstruct,'lambda1') ;    
        lambda1 = getfield(paramstruct,'lambda1') ; 
    end ;
    if isfield(paramstruct,'lambda2') ;    
        lambda2 = getfield(paramstruct,'lambda2') ; 
    end ;
end;

% specific exponential family functions
if strcmpi(distr1,'normal')
    fcn_db1=@(theta)(theta);
    fcn_ddb1=@(theta)ones(size(theta));
    fcn_dg1=@(mu)ones(size(mu));
%     lambda1=0; % override lambda1 with 0
elseif strcmpi(distr1,'binomial')
    fcn_db1=@(theta) exp(theta)./(1+exp(theta)); % adhoc bounded mean for binary to avoid degeneracy
    fcn_ddb1=@(theta) exp(theta)./((1+exp(theta)).^2); % adhoc bounded variance
    fcn_dg1=@(mu) 1./(mu.*(1-mu));
elseif strcmpi(distr1,'poisson')
    fcn_db1=@(theta)exp(theta);
    fcn_ddb1=@(theta)exp(theta);
    fcn_dg1=@(mu)1./mu;
%     lambda1=0;
end;   
if strcmpi(distr2,'normal')
    fcn_db2=@(theta)(theta);
    fcn_ddb2=@(theta)ones(size(theta));
    fcn_dg2=@(mu)ones(size(mu));
%     lambda2=0;
elseif strcmpi(distr2,'binomial')
    fcn_db2=@(theta) exp(theta)./(1+exp(theta));
    fcn_ddb2=@(theta) exp(theta)./((1+exp(theta)).^2);
    fcn_dg2=@(mu) 1./(mu.*(1-mu));
elseif strcmpi(distr2,'poisson')
    fcn_db2=@(theta)exp(theta);
    fcn_ddb2=@(theta)exp(theta);
    fcn_dg2=@(mu)1./mu;
%     lambda2=0;
end;   






% run for different ranks
for irun=1:length(rcand)
    r=rcand(irun);
    disp(['Running Rank ',num2str(r),':']);
    
    % initial value for EPCA
    rng(1234)
    Mu01=zeros(p1,1);
    Mu02=zeros(p2,1);
    if strcmpi(distr1,'poisson') 
        Mu01=mean(log(1+X1),1)'; 
    end;
    if strcmpi(distr2,'poisson') 
        Mu02=mean(log(1+X2),1)'; 
    end;    
    Mu0=[Mu01;Mu02];
    U0=randn(n,r); % total score, not just joint score, a bit of abuse of notation
    V0=GramSchmidt(randn(p1+p2,r));
    V01=V0(1:p1,:);
    V02=V0((p1+1):end,:);
    
    
    
    % est EPCA natural param from data with missing entries
    diff=inf;
    niter=0;

    rec=[];
    while diff>Tol && niter<Niter % iterate between U and V
        V0_old=V0;
        
        % estimate V01 Mu01
        for j=1:p1
            ValidEntry=find(MisInd1(:,j)==0);
            currY=X1(ValidEntry,j);
            currX=[ones(length(ValidEntry),1),U0(ValidEntry,:)];
            eta=currX*[Mu01(j),V01(j,:)]';
            mu=fcn_db1(eta);  
            sw=1./sqrt(fcn_ddb1(eta).*((fcn_dg1(mu)).^2)); 
            Yinduced=eta+(currY-mu).*fcn_dg1(mu);
            temp=wfitr(Yinduced,currX,sw,struct('lambda',lambda1));
            Mu01(j)=temp(1);
            V01(j,:)=temp(2:end);
        end;
        % estimate V02
        for j=1:p2
            ValidEntry=find(MisInd2(:,j)==0);
            currY=X2(ValidEntry,j);
            currX=[ones(length(ValidEntry),1),U0(ValidEntry,:)];
            eta=currX*[Mu02(j),V02(j,:)]';
            mu=fcn_db2(eta);
            sw=1./sqrt(fcn_ddb2(eta).*((fcn_dg2(mu)).^2)); 
            Yinduced=eta+(currY-mu).*fcn_dg2(mu);
            temp=wfitr(Yinduced,currX,sw,struct('lambda',lambda2));
            Mu02(j)=temp(1);
            V02(j,:)=temp(2:end);
        end;
        Mu0=[Mu01;Mu02];
        V0=[V01;V02];
% %         figure(1);clf;plot(Mu0);title('Mu');drawnow
% %         figure(2);clf;plot(V0);title('V');drawnow
        
        % estimate U0
        offset=Mu0*ones(1,n); % (p1+p2)*n offset matrix
        % calc weight
        eta1=offset(1:p1,:)+V01*U0'; % p1*n
        eta2=offset((p1+1):end,:)+V02*U0'; % p2*n
        mu1=fcn_db1(eta1);
        mu2=fcn_db2(eta2);
        W1=1./(fcn_ddb1(eta1).*((fcn_dg1(mu1)).^2)); % p1*n
        W2=1./(fcn_ddb2(eta2).*((fcn_dg2(mu2)).^2)); % p2*n
        sw=sqrt([W1;W2]);
        Yinduced=[V01*U0';V02*U0']+([tX1;tX2]-[mu1;mu2]).*[fcn_dg1(mu1);fcn_dg2(mu2)]; % (p1+p2)*n
        for i=1:n % update each row of U0 separately
            ValidEntry=find(tConMisInd(:,i)==0);
            tempu=wfitr(Yinduced(ValidEntry,i),V0(ValidEntry,:),sw(ValidEntry,i),struct('lambda',max(lambda1,lambda2)));
            U0(i,:)=tempu;
        end;        
%         figure(3);clf;plot(U0);title('U');drawnow
        
        
        
        % outer iteration stopping rule
        diff=PrinAngle(V0,V0_old); % max principal angle (0~90)
        niter=niter+1;
        

    end;

    
    
    if niter==Niter
        disp(['NOT converge after ',num2str(Niter),' iterations! Final PrinAngle=',num2str(diff)]);      
    else
        disp(['Converge after ',num2str(niter),' iterations.']);      
    end;


    % refit missing values to get CV scores
    Theta1=ones(n,1)*Mu01'+U0*V01';
    estMean1=fcn_db1(Theta1);
    Var1=fcn_ddb1(Theta1);
    X1miss=X1(MisInd1==1);
    X1pred=estMean1(MisInd1==1);
    X1var=Var1(MisInd1==1);
    Theta2=ones(n,1)*Mu02'+U0*V02';
    estMean2=fcn_db2(Theta2);
    Var2=fcn_ddb2(Theta2);
    X2miss=X2(MisInd2==1);
    X2pred=estMean2(MisInd2==1);
    X2var=Var2(MisInd2==1);
    

    
    % individual CV scores (Sum of squared Pearson residuals/num of missings)
    IndCVscore(1,irun)=sum(((X1miss-X1pred)./sqrt(X1var)).^2)/total_mis(1);
    IndCVscore(2,irun)=sum(((X2miss-X2pred)./sqrt(X2var)).^2)/total_mis(2);
    % combined CV scores (first convert to mean0 and std1)
    ComCVscore(irun)=(sum(((X1miss-X1pred)./sqrt(X1var)).^2)+...
        sum(((X2miss-X2pred)./sqrt(X2var)).^2))/sum(total_mis); % summation of two generalized Pearson Chisq stats
    
end;

end







    
function b = wfitr(y,X,sw,paramstruct)
% sw is a n*1 vector of sqrt weight matrix
% Perform a weighted least squares fit
% with ridge penalty with tuning parameter lambda
% |diag(sw)*y-diag(sw)*X*beta|^2 + n*lambda*|beta|^2

% default ridge parameter
lambda=1E-3;
if nargin > 3    
    lambda = getfield(paramstruct,'lambda') ; 
end;


[~,p] = size(X);
yw = y .* sw;
xw = bsxfun(@times,X, sw);
b = (xw'*xw+length(y)*lambda*eye(p))\(xw'*yw);
end