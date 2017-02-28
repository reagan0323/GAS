function [avgCVscore,rOpt,allCVscore]=Nfold_CV_Mixed(X1,X2,distr1,distr2,rcand,Nfold,paramstruct)
% This function calc N-fold CV scores for a mixed-type data set 
%
% Input
%     X1/X2         n*p1/n*p2 fully observed data matrix, from exponential family
%     distr1/distr2     string, specifying distribution,
%                       'normal','binomial','poisson'
%     rcand     a vector of candidate ranks for natural parameter matrix
%     Nfold     number of fold
%
%     paramstruct
%          seed    scalar, the random split seed, default is a fixed number
%          Niter   scalar, number of iterations per CV, default is 100
%          lambda1 lambda2  scalar, ridge parameter for glm, default is 0
% 
% Output
%     avgCVscore    a vector (same size with rcand) of CV scores, 
%                   each entry is the *mean* (can also use median) CV score across folds
%     rOpt          the optimal rank with the smallest CV score in rcand
%
%     allCVscore    Nfold*length(rcand) matrix, contains all CV scores, each row corresponds to a
%                   fold and each col corresponds to a tuning param
%
% need to call:
%    CV_mixEPCA_onestep1
%
% by Gen Li, 12/3/2016


% check
[n,p1]=size(X1);
[n0,p2]=size(X2);
if n~=n0
    error('Two data sets do not have matched samples!')
end;



seed=20160929; % for CV splitting
lambda1=0;
lambda2=0;
Niter=100;
if nargin > 6 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'seed') ;    
    seed = getfield(paramstruct,'seed') ; 
  end ;
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

% create blacklist
rng(seed);% set seed
if n*p1/Nfold-floor(n*p1/Nfold)~=0 
    warning('Number of CV folds is not a divisor of n*p1. Use random split...');
    leftoutnum=ceil(n*p1/Nfold); % number left out
    blacklist1=zeros(leftoutnum,Nfold);
    for i=1:Nfold
        blacklist1(:,i)=randsample(n*p1,leftoutnum);
    end;    
else
    blacklist1=reshape(randsample(n*p1,n*p1),n*p1/Nfold,Nfold); % each column corresp to the index of leftout samples in each fold, nonoverlap
end
if n*p2/Nfold-floor(n*p2/Nfold)~=0
    warning('Number of CV folds is not a divisor of n*p2. Use random split...');
    leftoutnum=ceil(n*p2/Nfold); % number left out
    blacklist2=zeros(leftoutnum,Nfold);
    for i=1:Nfold
        blacklist2(:,i)=randsample(n*p2,leftoutnum);
    end;    
else
    blacklist2=reshape(randsample(n*p2,n*p2),n*p2/Nfold,Nfold); % each column corresp to the index of leftout samples in each fold, nonoverlap
end
disp(['Leave out ',num2str(100/Nfold),'% entries for EPCA CV...']);




CVscore=[];
ifold=1;
while ifold<=Nfold % ntimes cross validation 
    disp([num2str(Nfold),'-Fold CV for mixEPCA, fold ',num2str(ifold)]);
    % omit X entries
    ind1 = blacklist1(:,ifold);
    ind2 = blacklist2(:,ifold);
    MisInd1=zeros(n,p1);MisInd1(ind1)=1;
    MisInd2=zeros(n,p2);MisInd2(ind2)=1;
    if sum(sum(MisInd1,1)==n)>0 || sum(sum(MisInd1,2)==p1)>0 || sum(sum(MisInd2,1)==n)>0 || sum(sum(MisInd2,2)==p2)>0
        warning('This fold contains missing rows or columns...skip...');
        ifold=ifold+1;
        continue
    else
        [ComCVscore,~,~]=CV_mixEPCA_onestep1(X1,X2,distr1,distr2,rcand,MisInd1,MisInd2,...
            struct('Niter',Niter,'lambda1',lambda1,'lambda2',lambda2));
        CVscore=[CVscore;ComCVscore];
        ifold=ifold+1;
    end;
end;

allCVscore=CVscore;
avgCVscore=median(CVscore,1); % or use median
[~,ind]=min(avgCVscore);
rOpt=rcand(ind);

end

