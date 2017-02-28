function [U,V]=ExpPCA(X,r,distr,paramstruct)
% This function implements the expoential family PCA (Collins, 2002)
% Alternating GLM (est all layers simultaneously, but no guarantee of 
% orthogonality in each iteration, so we post-process it to be orthogonal)
% Actually, it should be called exponential family SVD rather than PCA,
% because we do not adjust mean for data or constant predictor for natural param.
%
%
% input: 
%
%   X       n*p raw data matrix
%
%   r       scalar, rank of natural parameter matrix 
%
%   distr   string, distribution name, choose from 
%               'normal', default
%               'poisson',
%               'binomial', default is bernoulli
% 
%   paramstruct  struct('name',value)
%
%       Tol     stopping criterion based on likelihood, default 0.1
%       Niter   max number of iterations, default 200
%       lambda  ridge penalty parameter, default is 0, necessary for binomial
%       inner_Niter    max number of iterations for IRLS, default 20
%       inner_Tol      stopping criterion for IRLS based on likelihood, default 0.1
%
% Output: 
%
%   U       n*r score matrix of natural param matrix, absorbing D, orthogonal 
% 
%   V       p*r loading matrix of natural param matrix, orthonormal
%
%
% Note: this function estimates all ranks of U and V together, rather than 
%       cycling through different components as in Collins
%       For rank=1, no difference; for rank>1, may need to explore the
%       difference
% Note: In this version, we code the IRLS (with ridge) ourselves. For
%       simplicity, we let all rows to go through the same number of IRLS
%       iterations. Rigorously, each row should has its own stopping rule.
%       But since the number of iterations per IRLS does not really matter,
%       this is a simpler alternative.
%
% Created on 11/9/2016 by Gen Li


[n,p]=size(X);
if r>n || r>p
    error('Rank exceeds matrix dimension!')
end;

% default parameters
Tol=0.1; % stopping rule
Niter=200; % max iterations
lambda=0; % ridge parameter (for binomial, and sometimes for poisson)
inner_Niter=20; % default is full GLM for all rows in U and all ros in V (the number does not really matter that much)
inner_Tol=0.1;
if nargin > 3 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'Tol') ;    
    Tol = getfield(paramstruct,'Tol') ; 
  end ;
  if isfield(paramstruct,'Niter') ;    
    Niter = getfield(paramstruct,'Niter') ; 
  end ; 
  if isfield(paramstruct,'inner_Niter') ;    
    inner_Niter = getfield(paramstruct,'inner_Niter') ; 
  end ; 
  if isfield(paramstruct,'inner_Tol') ;    
    inner_Tol = getfield(paramstruct,'inner_Tol') ; 
  end ; 
  if isfield(paramstruct,'lambda') ;    
    lambda = getfield(paramstruct,'lambda') ; 
  end ; 
end;



switch distr
    case 'binomial'
        fcn_b=@(theta)log(1+exp(theta));
        fcn_db=@(theta)exp(theta)./(1+exp(theta)); % adhoc bounded mean for binary to avoid degeneracy
        fcn_ddb=@(theta)exp(theta)./((1+exp(theta)).^2); % adhoc bounded variance
        fcn_dg=@(mu)1./(mu.*(1-mu));
        %initial
        Prob=X;
        temp=rand(n,p)*0.1;
        Prob(X==1)=1-temp(X==1);
        Prob(X==0)=temp(X==0);
        [U,D,V]=svds(log(Prob./(1-Prob)),r);
        U=U*D;
    case 'poisson'
        fcn_b=@(theta)exp(theta);
        fcn_db=@(theta)exp(theta);
        fcn_ddb=@(theta)exp(theta);
        fcn_dg=@(mu)1./mu;
        % initial
        [U,D,V]=svds(log(X+1E-10),r);
        U=U*D;
    case 'normal'
        [U,D,V]=svds(X,r);
        U=U*D;
        disp(['Just SVD. Done!']);
        return;
end;


ThetaCurr=U*V';
temp=X.*ThetaCurr-fcn_b(ThetaCurr);
logl=sum(temp(:));

niter=1;
diff=inf;
rec=[]; % record likelihood
% alternative updating
while abs(diff)>Tol && niter<=Niter
    logl_prev=logl;
    U_old=U;
    V_old=V;
    
    
    % fix U, estimate V
    inner_iter=1;
    inner_diff=inf;
    ThetaCurr=U*V';
    temp=X.*ThetaCurr-fcn_b(ThetaCurr);
    logl=sum(temp(:));
    while inner_iter<=inner_Niter && inner_diff>inner_Tol % GLM
        logl_old=logl;
        eta=U*V';
        mu=fcn_db(eta);
        sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); 
        Yinduced=eta+(X-mu).*fcn_dg(mu);            
        parfor ind=1:p
            V(ind,:)=wfitr(Yinduced(:,ind),U,sw(:,ind),struct('lambda',lambda));
        end;
        ThetaCurr=U*V';
        temp=X.*ThetaCurr-fcn_b(ThetaCurr);
        logl=sum(temp(:));            
        inner_iter=inner_iter+1;
        inner_diff=abs(logl-logl_old);
        
%         rec=[rec,logl];
    end;
    if inner_iter==inner_Niter
        disp(['Inner iteration does NOT converge']);   
    end;


    % fix V, estimate U  
    inner_iter=1;
    inner_diff=inf;
    ThetaCurr=U*V';
    temp=X.*ThetaCurr-fcn_b(ThetaCurr);
    logl=sum(temp(:));
    while inner_iter<=inner_Niter && inner_diff>inner_Tol % GLM
        logl_old=logl;
        eta=V*U';
        mu=fcn_db(eta);
        sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); 
        Yinduced=eta+(X'-mu).*fcn_dg(mu);            
        parfor ind=1:n
            U(ind,:)=wfitr(Yinduced(:,ind),V,sw(:,ind),struct('lambda',lambda));
        end;
        ThetaCurr=U*V';
        temp=X.*ThetaCurr-fcn_b(ThetaCurr);
        logl=sum(temp(:));            
        inner_iter=inner_iter+1;
        inner_diff=abs(logl-logl_old);
%         rec=[rec,logl];
    end;
    if inner_iter==inner_Niter
        disp(['Inner iteration does NOT converge']);      
    end;

    
    
    % orthogonalize
    [U,D,V]=svds(U*V',r);
    U=U*D;
    asign=sign(U(1,:));
    U=bsxfun(@times,U,asign);
    V=bsxfun(@times,V,asign);

    
    % stopping rule
    ThetaCurr=U*V';
    temp=X.*ThetaCurr-fcn_b(ThetaCurr);
    logl=sum(temp(:));
    rec=[rec,logl];
    
    diff=(logl-logl_prev);
    niter=niter+1;   
    figure(100);clf;plot(rec,'o-');title(['log likelihood (diff=',num2str(diff),')']);drawnow;    

end;

if niter==Niter
    disp([distr,' SVD does NOT converge after ',num2str(Niter),' iterations! Final angle change is ',num2str(diff)]);      
else
    disp([distr,' SVD converges after ',num2str(niter),' iterations.']);      
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

    
    
    


