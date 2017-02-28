function [U0,U1,U2,V0,A1,A2,Mean0,flag]=GAS(X1,X2,r0,r1,r2,distr1,distr2,paramstruct)
% This function can be used for fitting GAS between two (non-Gaussian)
% data sets, with or without sparsity in the joint loading.
% The GAS model is 
%       Theta1=ones(n,1)*Mean1'+U0*V1'+U1*A1'
%       Theta2=ones(n,1)*Mean2'+U0*V2'+U2*A2'
% where Mean0=[Mean1;Mean2] and V0=[V1;V2].
%
% Overall, we use an alternating algorithm to cycle through joint loading,
% joint score, individual loadings, and individual scores. Each step is a
% GLM problem (with offset, with or without intercept). By default, we use 
% a one-step approximation of IRLS with warm start for each GLM instead of
% the full inner iteration. When necessary, we also add a small ridge
% penalty term to the GLM problem to avoid singularity.
% 
% Parameters satify the identifiability conditions in the paper.
% 1. U0 orth U1, U0 orth U2, U1 not orth U2
% 2. V0, A1, A2 orthornormal separately, but no mutual orthogonality
%
% Input: 
%
%   X1      n*p1 data matrix from distr1 
%
%   X2      n*p2 data matrix from distr2 
%
%   r0      scalar, joint rank (excluding the intercept)
%
%   r1      scalar, individual rank for Theta1  (excluding the intercept)
%   
%   r2      scalar, individual rank for Theta2  (excluding the intercept)
%
%   distr1   string, choose from 'normal' 'binomial' 'poisson'
%
%   distr2   string, choose from 'normal' 'binomial' 'poisson'
%
%
%   paramstruct
%       
%       Niter       number of overall iteration, default 500
%
%       Tol         Threshold for likelihood difference, default 0.1
%
%       sparsity    0: dense V0; 1: sparse V0, default 0 (dense) 
%                   Note: use FIT-SSVD by Yang et al, no need to choose
%                   sparsity tuning parameter.
%
%       inner_Niter     number of inner iterations for every GLM problem,
%                       default 1 (ie, one-step approximation)
%                       Note: if large, the overall algorithm is guaranteed to
%                       converge, but may need non-zero ridge parameter for
%                       binomial data.
%
%       lambda1     ridge parameter for X1 structures, default 0
%                   (Recommended to be 1E-3, if inner_Niter>1 and
%                   distr1='binomial')
%
%       lambda2     ridge parameter for X2 structures, default 0
%                   (Recommended to be 1E-3, if inner_Niter>1 and
%                   distr2='binomial')
%
%       initial     struct, initial values for output parameters, contains
%                   U0,U1,U2,V0,A1,A2,Mean0. Default is random start
%             
%       thres_method    string, 'hard' (default) or 'soft' for SPCA. Only
%                       useful when sparsity=1
%       fig         debug tool, 0 (default) means no interruption; 1 means
%                      output figures in each iteration
% Output: 
%
%   U0      n*r0 joint score matrix for the natural parameter matrices
%           with orthogonal columns, and orthogonal with U1 and U2
% 
%   U1      n*r1 individual score matrix for the natural parameter matrix
%           of X1, with orthogonal columns, and orthogonal with U0
%
%   U1      n*r2 individual score matrix for the natural parameter matrix
%           of X2, with orthogonal columns, and orthogonal with U0
% 
%   V0      (p1+p2)*r0 joint loading matrix, with orthonormal columns. 
%           The top p1*r0 submatrix is for X1 and the bottom p2*r0 is for X2
%           If sparsity=1, this is a sparse matrix
%
%   A1       p1*r1 individual loading matrix, with orthonormal columns (no sparsity)
%          
%   A2       p2*r2 individual loading matrix, with orthonormal columns (no sparsity)
%
%   Mean0    (p1+p2)*1 Mean vector for the natural parameters 
%
%   flag     0 or 1, 0 means convergence, 1 means non-convegence.
%
%
% Created by Gen Li, 11/18/2016



% check
[n,p1]=size(X1);
[n2,p2]=size(X2);
tX1=X1';
tX2=X2';
if n2~=n 
    error('Mismatched samples!');
end;
if r0+r1>min(n,p1) || r0+r2>min(n,p2)
    error('Rank too large!');
end;



%% default parameters
sparsity=0; % default: dense joint loading
Niter=500; % max iterations
Tol=0.1; % stopping rule 
inner_Niter=1; % default: onestep
lambdaridge1=0; % default: no ridge penalty
lambdaridge2=0; % default: no ridge penalty
thres_method='hard'; % default: when use SPCA, use hard thresholding
fig=0;
% initial estimate
if nargin > 7 && isfield(paramstruct,'initial')   % initial values  
    temp = paramstruct.initial ;
    U0=temp.U0;
    U1=temp.U1;
    U2=temp.U2;
    V0=temp.V0;
    A1=temp.A1;
    A2=temp.A2;
    Mean0=temp.Mean0;
else
    rng(1234)
    if r0~=0
        U0=GramSchmidt(randn(n,r0));
        V0=GramSchmidt(randn(p1+p2,r0));
    else
        U0=zeros(n,1);
        V0=zeros(p1+p2,1);
    end;
    if r1~=0
        U1=GramSchmidt(randn(n,r1));
        A1=GramSchmidt(randn(p1,r1));
    else 
        U1=zeros(n,1);
        A1=zeros(p1,1);
    end;
    if r2~=0
        U2=GramSchmidt(randn(n,r2));
        A2=GramSchmidt(randn(p2,r2));
    else 
        U2=zeros(n,1);
        A2=zeros(p2,1);
    end;
    Mean0=zeros(p1+p2,1);
end;
Mean1=Mean0(1:p1);
Mean2=Mean0((p1+1):end);
%
if nargin > 7 % other values
    if isfield(paramstruct,'Niter') ;
        Niter = paramstruct.Niter ; 
    end ;
    if isfield(paramstruct,'Tol') ;    
        Tol = paramstruct.Tol ; 
    end ;
    if isfield(paramstruct,'sparsity') ;    
        sparsity = paramstruct.sparsity ;
    end ; 
    if isfield(paramstruct,'thres_method') ;    
        thres_method = paramstruct.thres_method ;
    end ;
    if isfield(paramstruct,'inner_Niter') ;    
        inner_Niter = paramstruct.inner_Niter ; 
    end ;
    if isfield(paramstruct,'lambda1') ;    
        lambdaridge1 = paramstruct.lambda1 ; 
    end ;
    if isfield(paramstruct,'lambda2') ;    
        lambdaridge2 = paramstruct.lambda2 ; 
    end ;
    if isfield(paramstruct,'fig') ;    
        fig = paramstruct.fig ; 
    end ;
end;


% define critical functions
switch distr1
    case 'binomial'
        fcn_b1=@(theta)log(1+exp(theta));
        fcn_db1=@(theta)exp(theta)./(1+exp(theta)); % adhoc bounded mean for binary to avoid degeneracy
        fcn_ddb1=@(theta)exp(theta)./((1+exp(theta)).^2); % adhoc bounded variance
        fcn_dg1=@(mu)1./(mu.*(1-mu));
    case 'poisson'
        fcn_b1=@(theta)exp(theta);
        fcn_db1=@(theta)exp(theta);
        fcn_ddb1=@(theta)exp(theta);
        fcn_dg1=@(mu)1./mu;
        Mean1=mean(log(1+X1),1)'; % for poisson, override mu to reduce crash
    case 'normal'
        fcn_b1=@(theta)(theta.^2)/2;
        fcn_db1=@(theta)theta;
        fcn_ddb1=@(theta)ones(size(theta));
        fcn_dg1=@(mu)ones(size(mu));
end;
switch distr2
    case 'binomial'
        fcn_b2=@(theta)log(1+exp(theta));
        fcn_db2=@(theta)exp(theta)./(1+exp(theta));
        fcn_ddb2=@(theta)exp(theta)./((1+exp(theta)).^2);
        fcn_dg2=@(mu)1./(mu.*(1-mu));
    case 'poisson'
        fcn_b2=@(theta)exp(theta);
        fcn_db2=@(theta)exp(theta);
        fcn_ddb2=@(theta)exp(theta);
        fcn_dg2=@(mu)1./mu;
        Mean2=mean(log(1+X2),1)';
    case 'normal'
        fcn_b2=@(theta)(theta.^2)/2;
        fcn_db2=@(theta)theta;
        fcn_ddb2=@(theta)ones(size(theta));
        fcn_dg2=@(mu)ones(size(mu));
end;
Mean0=[Mean1;Mean2];

% calc initial likelihood
combTheta=ones(n,1)*Mean0'+U0*V0'+[U1*A1',U2*A2'];
Theta1=combTheta(:,1:p1);
Theta2=combTheta(:,(p1+1):end);
temp=[X1.*Theta1-fcn_b1(Theta1) , X2.*Theta2-fcn_b2(Theta2)];
logl=sum(temp(:));
   


%% Main Iteration
rec=zeros(Niter+1,1);rec(1)=logl; % initial log likelihood
diff=inf;
niter=0;
flag=0; % converge
while abs(diff)>Tol && niter<Niter
    logl_old=logl;
    V1=V0(1:p1,:);
    V2=V0((p1+1):end,:);
    Mean1=Mean0(1:p1);
    Mean2=Mean0((p1+1):end);

%     % check est updates in each iteration
    if fig
    figure(1);clf;plot(abs(V0));ylim([0,0.5]);title('V0');drawnow
%     figure(2);clf;plot(abs(Mean0));title('Mean0');drawnow
%     figure(3);clf;plot(abs(U0));title('U0');drawnow
%     figure(4);clf;plot(abs(A1));title('A1');drawnow
%     figure(5);clf;plot(abs(U1));title('U1');drawnow
%     figure(6);clf;plot(abs(A2));title('A2');drawnow
%     figure(7);clf;plot(abs(U2));title('U2');drawnow
    end;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % estimate individual structures %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    offset1=U0*V1';
    offset2=U0*V2';  
    toffset1=offset1';
    toffset2=offset2';   
    % Est A1&Mean1 and U1 (no sparsity)
    if r1~=0
    for iter=1:inner_Niter
        eta1=[ones(n,1),U1]*[Mean1';A1']+offset1;
        mu1=fcn_db1(eta1);
        sw=1./sqrt(fcn_ddb1(eta1).*((fcn_dg1(mu1)).^2)); 
        Yinduced=(eta1-offset1)+(X1-mu1).*fcn_dg1(mu1);
        parfor ind=1:p1
            temp=wfitr(Yinduced(:,ind),[ones(n,1),U1],sw(:,ind),struct('lambda',lambdaridge1));
            Mean1(ind)=temp(1);
            A1(ind,:)=temp(2:end);
        end; 
    end;
    for iter=1:inner_Niter
        eta1=A1*U1'+toffset1+Mean1*ones(1,n); 
        mu1=fcn_db1(eta1);
        sw=1./sqrt(fcn_ddb1(eta1).*((fcn_dg1(mu1)).^2)); 
        Yinduced=(A1*U1')+(X1'-mu1).*fcn_dg1(mu1);
        parfor ind=1:n
            temp=wfitr(Yinduced(:,ind),A1,sw(:,ind),struct('lambda',lambdaridge1));
            U1(ind,:)=temp;
        end;    
    end;
    end;
    % Est A2&Mean2 and U2 (no sparsity)   
    if r2~=0
    for iter=1:inner_Niter
        eta2=[ones(n,1),U2]*[Mean2';A2']+offset2;
        mu2=fcn_db2(eta2);
        sw=1./sqrt(fcn_ddb2(eta2).*((fcn_dg2(mu2)).^2)); 
        Yinduced=(eta2-offset2)+(X2-mu2).*fcn_dg2(mu2);
        parfor ind=1:p2
            temp=wfitr(Yinduced(:,ind),[ones(n,1),U2],sw(:,ind),struct('lambda',lambdaridge2));
            Mean2(ind)=temp(1);
            A2(ind,:)=temp(2:end);
        end;
    end;
    for iter=1:inner_Niter
        eta2=A2*U2'+toffset2+Mean2*ones(1,n); 
        mu2=fcn_db2(eta2);
        sw=1./sqrt(fcn_ddb2(eta2).*((fcn_dg2(mu2)).^2)); 
        Yinduced=(A2*U2')+(X2'-mu2).*fcn_dg2(mu2);
        parfor ind=1:n
            temp=wfitr(Yinduced(:,ind),A2,sw(:,ind),struct('lambda',lambdaridge2));
            U2(ind,:)=temp;
        end; 
    end;
    end;
  
  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % estimate JOINT structures %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Est V0 & Mean0 (no sparsity yet)
    if r0~=0
    offset1=U1*A1';
    offset2=U2*A2'; 
    for iter=1:inner_Niter
        eta1=[ones(n,1),U0]*[Mean1';V1']+offset1;
        mu1=fcn_db1(eta1);
        sw=1./sqrt(fcn_ddb1(eta1).*((fcn_dg1(mu1)).^2)); 
        Yinduced=(eta1-offset1)+(X1-mu1).*fcn_dg1(mu1);
        parfor ind=1:p1
            temp=wfitr(Yinduced(:,ind),[ones(n,1),U0],sw(:,ind),struct('lambda',lambdaridge1));
            Mean1(ind)=temp(1);
            V1(ind,:)=temp(2:end);
        end;
    end;
    for iter=1:inner_Niter
        eta2=[ones(n,1),U0]*[Mean2';V2']+offset2;
        mu2=fcn_db2(eta2);
        sw=1./sqrt(fcn_ddb2(eta2).*((fcn_dg2(mu2)).^2)); 
        Yinduced=(eta2-offset2)+(X2-mu2).*fcn_dg2(mu2);
        parfor ind=1:p2
            temp=wfitr(Yinduced(:,ind),[ones(n,1),U0],sw(:,ind),struct('lambda',lambdaridge2));
            Mean2(ind)=temp(1);
            V2(ind,:)=temp(2:end);
        end;
    end;
    Mean0=[Mean1;Mean2];
    V0=[V1;V2];
    
    
    
    % Est Joint score U0 
    offset1=U1*A1'+ones(n,1)*Mean1';
    offset2=U2*A2'+ones(n,1)*Mean2';     
    for iter=1:inner_Niter
        eta1=V1*U0'+offset1';
        eta2=V2*U0'+offset2';
        mu1=fcn_db1(eta1);
        mu2=fcn_db2(eta2);
        W1=1./(fcn_ddb1(eta1).*((fcn_dg1(mu1)).^2)); % diagonal of weight matrix for IRLS
        W2=1./(fcn_ddb2(eta2).*((fcn_dg2(mu2)).^2));
        sw=sqrt([W1;W2]);
        Yinduced=([eta1;eta2]-[offset1';offset2'])+([tX1;tX2]-[mu1;mu2]).*[fcn_dg1(mu1);fcn_dg2(mu2)];
        parfor ind=1:n
            tempu=wfitr(Yinduced(:,ind),V0,sw(:,ind),struct('lambda',max(lambdaridge1,lambdaridge2)));
            U0(ind,:)=tempu;
        end;        
    end;
    end;



    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % normalize model parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if r0~=0
    tempU0=[ones(n,1),U0];
    Proj=tempU0/(tempU0'*tempU0)*tempU0'; % make U1 and U2 orthogonal to U0
    cProj=(eye(n)-Proj);
    [newU1,newD1,newA1]=svds(cProj*U1*A1',r1);
    newU1=newU1*newD1;
    [newU2,newD2,newA2]=svds(cProj*U2*A2',r2);
    newU2=newU2*newD2;
    newJoint=tempU0*[Mean0,V0]'+[Proj*U1*A1',Proj*U2*A2'];
    Mean0=mean(newJoint,1)';
    dmJoint=bsxfun(@minus,newJoint,Mean0');
    if sparsity==0 % dense
        [newU0,newD0,newV0]=svds(dmJoint,r0);
        newU0=newU0*newD0;  
        U0=newU0;
        V0=newV0;    
    else % sparse
        [U0,V0]=SPCA_custom(dmJoint(:,1:p1),dmJoint(:,(p1+1):end),r0,struct('thres_method',thres_method)); 
    end;
    U1=newU1;
    U2=newU2;
    A1=newA1;
    A2=newA2;
    end;
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate stopping rule
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    niter=niter+1;
    % log likelihood
    combTheta=ones(n,1)*Mean0'+U0*V0'+[U1*A1',U2*A2'];
    Theta1=combTheta(:,1:p1);
    Theta2=combTheta(:,(p1+1):end);
    temp=[X1.*Theta1-fcn_b1(Theta1) , X2.*Theta2-fcn_b2(Theta2)];
    logl=sum(temp(:));
    diff=logl-logl_old;
    % check
    rec(niter+1)=logl; 
    
    if fig
    figure(100);clf; 
    plot(0:niter,rec(1:(niter+1)),'o-');
    title(['Log likelihood (diff=',num2str(diff),')']);
    drawnow; 
    end;
end;


if niter==Niter
    disp(['GAS does NOT converge after ',num2str(Niter),' iterations! Final loglik change is ',num2str(diff)]); 
    flag=1;
else
    disp(['GAS converges after ',num2str(niter),' iterations.']);      
end;

end




    
function b = wfitr(y,X,sw,paramstruct)
% sw is a n*1 vector of sqrt weight matrix
% Perform a weighted least squares fit
% with ridge penalty with tuning parameter lambda
%
% |diag(sw)*y-diag(sw)*X*beta|^2 + n*lambda*|beta|^2

% default ridge parameter
lambda=1E-3;
if nargin > 3    
    lambda = paramstruct.lambda ; 
end;

[~,p] = size(X);
yw = y .* sw;
xw = bsxfun(@times,X, sw);
b = (xw'*xw+length(y)*lambda*eye(p))\(xw'*yw);
end




function [U,V]=SPCA_custom(X1,X2,r,paramstruct)
% This file conduct rank-r sparse PCA for data matrix X using an alternating algorithm. 
% Reference: Yang, MA, Buja, 2013, JCGS
% 

hard=1; % default is hard-thresholding
if nargin > 3 % other values
  if isfield(paramstruct,'thres_method') ;    
    temp = paramstruct.thres_method ;  % range of elastic net tuning, meaningless when sparsity=0
    if strcmp(temp,'soft')
        hard=0;
    else
        hard=1;
    end;
  end ;
end;
% initial values
[~,p1]=size(X1);
[~,p2]=size(X2);
sigma1=1.4826*mad(X1(:));
sigma2=1.4826*mad(X2(:));
lam1=sigma1*sqrt(2*log(p1)); % asymptotic threshold for V
lam2=sigma2*sqrt(2*log(p2));
[U,~,V]=svds([X1,X2],r);

diff=inf;
Tol=0.1; % threshold for max principal angles between two estimates
niter=1;
Niter=200;
while diff>Tol && niter<=Niter
    V_old=V;
    % update V
    V1=X1'*U;
    V2=X2'*U;
    if hard % hard thresholding
        V1(abs(V1)<lam1)=0;
        V2(abs(V2)<lam2)=0;
    else % soft thresholding
        V1(V1>0)=max(V1(V1>0)-lam1,0);
        V1(V1<0)=min(V1(V1<0)+lam1,0);
        V2(V2>0)=max(V2(V2>0)-lam2,0);
        V2(V2<0)=min(V2(V2<0)+lam2,0);
    end;
    V=[V1;V2];
    [V,~]=qr(V,0); % orthonormal col in U
    
    % update U (no need for sparsity)
    U=[X1,X2]*V;
    [U,~]=qr(U,0); % orthonormal col in U

    diff=180/pi*acos(min(svd(V'*V_old,'econ'))); % max Prin Angle for the two orthonormal matrices
    niter=niter+1;
end;
if niter==Niter
    disp(['SPCA does NOT converge after ',num2str(Niter),' iterations!']);      
end;

% the final V has orthonomal columns
[V,~]=qr(V,0);
U=[X1,X2]*V;
end
