function [avgCVscore,rOpt,allCVscore]=Nfold_CV_Single(X,distr,rcand,Nfold,paramstruct)
% This function calc N-fold CV scores for an EPCA problem for a set of ranks 
%
% Input
%     X         n*p fully observed data matrix, from exponential family
%               or 1*2 cell array for binomial distribution {NumSuccess,NumTrial}
%     distr     string, specifying distribution
%               'normal','poisson','binomial'
%     rcand     a vector of candidate ranks for natural parameter matrix
%     Nfold     number of fold
%
%     paramstruct
%          seed    scalar, the random split seed, default is a fixed number
%          Niter   scalar, number of iterations per CV, default is 100
%          lambda  scalar, ridge parameter for glm, default is 0
% 
% 
% Output
%     avgCVscore    a vector (same size with rcand) of CV scores, 
%                   each entry is avg (x-hat{x})^2 across folds
%     rOpt          the optimal rank with the smallest CV score in rcand (selected by median)
%
%     allCVscore    Nfold*length(rcand) matrix, contains all CV scores, each row corresponds to a
%                   fold and each col corresponds to a tuning param
%
% need to call:
%    CV_EPCA_onestep1
%
% by Gen Li, 12/3/2016


seed=20161225;
lambda=0;
Niter=100;
if nargin > 4 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'seed') ;    
    seed = getfield(paramstruct,'seed') ; 
  end ;
    if isfield(paramstruct,'Niter') ;    
        Niter = getfield(paramstruct,'Niter') ; 
    end ;
    if isfield(paramstruct,'lambda') ;    
        lambda = getfield(paramstruct,'lambda') ; 
    end ;
end;
rng(seed);% set seed



disp(['Leave out ',num2str(100/Nfold),'% entries for EPCA CV...']);

% split data into Nfold 
[n,p]=size(X);
% check
if n*p/Nfold-floor(n*p/Nfold)~=0
    warning('Number of CV folds is not a divisor of n*p. Will randomly split...');
    leftoutnum=ceil(n*p/Nfold); % number left out
    blacklist=zeros(leftoutnum,Nfold);
    for i=1:Nfold
        blacklist(:,i)=randsample(n*p,leftoutnum);
    end;
else % non-overlap splitting
    blacklist=reshape(randsample(n*p,n*p),n*p/Nfold,Nfold); % each column corresp to the index of leftout samples in each fold, nonoverlap
end;


CVscore=[];
ifold=1;
while ifold<=Nfold % ntimes cross validation 
    disp(['Fold ',num2str(ifold)]);
    % creat sparse data for this fold
    ind = blacklist(:,ifold);
    MisInd=zeros(n,p);
    MisInd(ind)=1; % missing index, 1=missing

    if sum(sum(MisInd,1)==n)>0 || sum(sum(MisInd,2)==p)>0
        warning('This fold contains missing rows or columns...skip...');
        ifold=ifold+1;
        continue
    else
        [score,~]=CV_EPCA_onestep1(X,distr,rcand,MisInd,struct('Niter',Niter,'lambda',lambda));
        CVscore=[CVscore;score];
        ifold=ifold+1;
    end;
end;
allCVscore=CVscore;
avgCVscore=median(CVscore,1); % this avg is ok
[~,ind]=min(avgCVscore);
rOpt=rcand(ind);







