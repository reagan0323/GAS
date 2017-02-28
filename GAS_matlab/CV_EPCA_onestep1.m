function [CVscore,total_mis]=CV_EPCA_onestep1(X,distr,rcand,MisInd,paramstruct)
% This function calc CV scores for an EPCA problem for a set of ranks 
% in one realization of multiple missing observations (indicated by MisInd). 
%
% Input
%     X         n*p fully observed data matrix, from exponential family
%     distr     string, specifying distribution 'normal','binomial','poisson'
%     rcand     a vector of candidate ranks for natural parameter matrix
%     MisInd    n*p 0/1 matrix, corresponding to X, 1=missing
%
% Output
%     CVscore       a vector of CV scores corresp to rcand, each entry is 
%                   sum (x_i-hat{x_i})^2 /N_i, where i is the index of MisInd=1
%     total_mis     total number of missing entries
%
%
% by Gen Li, 11/8/2016

[n,p]=size(X);
[n_,p_]=size(MisInd);
if n_~=n || p_~=p
    error('Missing Index matrix and Data matrix not compatible!');
end;
CVscore=zeros(size(rcand));
total_mis=sum(sum(MisInd));

% lambda=1E-3; % default
lambda=0;
Niter=100; % default
if nargin > 4 % other values
    if isfield(paramstruct,'Niter') ;    
        Niter = getfield(paramstruct,'Niter') ; 
    end ;
    if isfield(paramstruct,'lambda') ;    
        lambda = getfield(paramstruct,'lambda') ; 
    end ;
end;

% specific exponential family functions
switch distr
    case 'binomial'
        fcn_db=@(theta)exp(theta)./(1+exp(theta));
        fcn_ddb=@(theta)exp(theta)./((1+exp(theta)).^2);
        fcn_dg=@(mu)1./(mu.*(1-mu));
    case 'poisson'
        fcn_db=@(theta)exp(theta);
        fcn_ddb=@(theta)exp(theta);
        fcn_dg=@(mu)1./mu;
    case 'normal'
        fcn_db=@(theta)theta;
        fcn_ddb=@(theta)ones(size(theta));
        fcn_dg=@(mu)ones(size(mu));
end;



% run for different ranks
for irun=1:length(rcand)
    r=rcand(irun);
    disp(['Running Rank ',num2str(r),':']);
    
    % initial value for EPCA
    if strcmpi(distr,'poisson') % cheat a bit in initial values
        Mu=mean(log(X+1),1)';
        [U,D,V]=svds(log(X+1)-ones(n,1)*Mu',r);
        U=U*D;
    else
        rng(123);
        U=randn(n,r);
        V=GramSchmidt(randn(p,r));  
        Mu=zeros(p,1);
    end;

    % perform EPCA with missing entries
    diff=inf;
    niter=0;
    % rec=[];
    while diff>0.1 && niter<Niter
%         niter
        V_old=V;
        % estimate V and Mu
        for j=1:p
            ValidEntry=find(MisInd(:,j)==0);
            currY=X(ValidEntry,j);
            currX=[ones(length(ValidEntry),1),U(ValidEntry,:)];
            eta=currX*[Mu(j),V(j,:)]';
            mu=fcn_db(eta);
            sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); 
            Yinduced=eta+(currY-mu).*fcn_dg(mu);
            temp=wfitr(Yinduced,currX,sw,struct('lambda',lambda));
            Mu(j)=temp(1);
            V(j,:)=temp(2:end);
        end;
%         figure(1);clf;plot(V);title('V');drawnow
        
        % estimate each row of U
        for i=1:n
            ValidEntry=find(MisInd(i,:)==0);
            offset=Mu(ValidEntry);
            currY=X(i,ValidEntry)';
            currX=V(ValidEntry,:);
            eta=offset+currX*U(i,:)';
            mu=fcn_db(eta);
            sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); 
            Yinduced=(eta-offset)+(currY-mu).*fcn_dg(mu);
            U(i,:)=wfitr(Yinduced,currX,sw,struct('lambda',lambda)); 
        end;
%         figure(2);clf;plot(U);title('U');drawnow
        

        diff=PrinAngle(V,V_old); % max principal angle (0~90)
        niter=niter+1;
    end;

    if niter==Niter
        disp(['NOT converge after ',num2str(Niter),' iterations! Final PrinAngle=',num2str(diff)]);      
    else
        disp(['Converge after ',num2str(niter),' iterations.']);      
    end;


    % refit missing values
    Theta=ones(n,1)*Mu'+U*V';
    estMean=fcn_db(Theta);
    Xtrue=X(MisInd==1);
    Xpred=estMean(MisInd==1);
    CVscore(irun)=ExpMetric(Xtrue,Xpred,distr);
end;

end





function b = wfitr(y,X,sw,paramstruct)
% sw is a n*1 vector of sqrt weight matrix
% Perform a weighted least squares fit
% with ridge penalty with tuning parameter lambda
% |diag(sw)*y-diag(sw)*X*beta|^2 + n*lambda*|beta|^2

% default ridge parameter
lambda=1E-4;
if nargin > 3    
    lambda = getfield(paramstruct,'lambda') ; 
end;


[~,p] = size(X);
yw = y .* sw;
xw = bsxfun(@times,X, sw);
b = (xw'*xw+length(y)*lambda*eye(p))\(xw'*yw);
end
    


function out=ExpMetric(Xtrue,Xpred,distr)
% this function calc the "sum of squared pearson residuals divided by the num, ie, MSE (GL 1/6/2017)"
% Can accommodate normal, poisson, and bernoulli
n1=length(Xtrue(:));
n2=length(Xpred(:));
Xtrue=Xtrue(:);
Xpred=Xpred(:);

if n1~=n2
    error('Not equal size...');
end;

if strcmpi(distr,'normal') 
    out=sum((Xtrue-Xpred).^2)/n1;
elseif strcmpi(distr,'poisson')
    out=sum(((Xtrue-Xpred)./sqrt(Xpred)).^2)/n1;
elseif strcmpi(distr,'binomial')
    out=sum(((Xtrue-Xpred)./sqrt(Xpred.*(1-Xpred))).^2)/n1; % MSE
else
    error('No such distribution...');
end;


end