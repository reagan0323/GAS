% This file contains all simulation settings for GAS simulation study
%  by Gen Li, 11/16/2016

clc
close all
addpath GAS_matlab

%% simulation settings
% Basic settings
% dimension and rank
rng(20160926);
n=200;
p1=120;
p2=120;
p=[p1,p2];
r0=2; % true rank, excluding the intercept
r1=2;
r2=2;
r=[r0,r1,r2];
% U
temp=rand(n,r0+r1+r2)-0.5;
temp=bsxfun(@minus,temp,mean(temp,1));
U=GramSchmidt(temp);  % score vectors are centered and orthonormal
U0=U(:,1:r0);
U1=U(:,(r0+1):(r0+r1));
U2=U(:,(r0+r1+1):end);


switch choosesetting
    case 1 % normal normal (JIVE equivalent)
        distr1='normal';
        distr2='normal';

        % V
        V0true=GramSchmidt(randn(p1+p2,r0)); % equal weight for two data sets
        V1true=V0true(1:p1,:);
        V2true=V0true((p1+1):end,:);
        A1true=GramSchmidt(rand(p1,r1)-0.5);
        A2true=GramSchmidt(rand(p2,r2)-0.5);
        % D
        D0true=[180,140];U0true=U0*diag(D0true);
        D1true=[120,100];U1true=U1*diag(D1true);
        D2true=[100,80];U2true=U2*diag(D2true);
        % Mu
        Mu0true=rand(p1+p2,1)-0.5;
        Mu1true=Mu0true(1:p1,:);
        Mu2true=Mu0true((p1+1):end,:);

    case 2 % normal binomial
        distr1='normal';
        distr2='binomial';
        % V
        V0true=GramSchmidt([(rand(p1,r0)-0.5);2*(rand(p2,r0)-0.5)]); % (binary needs stronger signal)
        V1true=V0true(1:p1,:);
        V2true=V0true((p1+1):end,:);
        A1true=GramSchmidt(rand(p1,r1)-0.5);
        A2true=GramSchmidt(rand(p2,r2)-0.5);
        % D
        D0true=[240,220];U0true=U0*diag(D0true);
        D1true=[90,80];U1true=U1*diag(D1true);
        D2true=[200,180];U2true=U2*diag(D2true);
        % Mu
        Mu0true=rand(p1+p2,1)-0.5;
        Mu1true=Mu0true(1:p1,:);
        Mu2true=Mu0true((p1+1):end,:);
        
    case 3 % normal poisson
        distr1='normal';
        distr2='poisson';
        % V
        V0true=GramSchmidt([(rand(p1,r0)-0.5);0.5*(rand(p2,r0)-0.5)]); % (poisson needs weaker signal)
        V1true=V0true(1:p1,:);
        V2true=V0true((p1+1):end,:);
        A1true=GramSchmidt(rand(p1,r1)-0.5,V1true);
        A2true=GramSchmidt(rand(p2,r2)-0.5,V2true);
        % D
        D0true=[80,40];U0true=U0*diag(D0true);
        D1true=[60,40];U1true=U1*diag(D1true);
        D2true=[20,16];U2true=U2*diag(D2true);
        % Mu
        Mu0true=[rand(p1,1)-0.5;2+rand(p2,1)];
        Mu1true=Mu0true(1:p1,:);
        Mu2true=Mu0true((p1+1):end,:);
       
    case 4 % binomial poisson (hardest)
        distr1='binomial';
        distr2='poisson';
        % V
        V0true=GramSchmidt([10*(rand(p1,r0)-0.5);rand(p2,r0)-0.5]); % (binary needs stronger signal)
        V1true=V0true(1:p1,:);
        V2true=V0true((p1+1):end,:);
        A1true=GramSchmidt(rand(p1,r1)-0.5,V1true);
        A2true=GramSchmidt(rand(p2,r2)-0.5,V2true);
        % D
        D0true=[180,140];U0true=U0*diag(D0true);
        D1true=[200,160];U1true=U1*diag(D1true);
        D2true=[12,10];U2true=U2*diag(D2true);
        % Mu
        Mu0true=[rand(p1,1)-0.5;2+rand(p2,1)];
        Mu1true=Mu0true(1:p1,:);
        Mu2true=Mu0true((p1+1):end,:);
end;
if sparsity==1 % sparse
    V0true=GramSchmidt([hard_thres(V1true,quantile(abs(V1true(:)),0.4));hard_thres(V2true,quantile(abs(V2true(:)),0.4))]); 
    V1true=V0true(1:p1,:); sum(V1true==0,1)/size(V1true,1)
    V2true=V0true((p1+1):end,:); sum(V2true==0,1)/size(V2true,1)
    simname=[distr1,'_',distr2,'_sparse'];
else
    simname=[distr1,'_',distr2];
end;
disp(['The simulation setting is ',simname])




% signal
Mean1true=ones(n,1)*Mu1true';
Mean2true=ones(n,1)*Mu2true';
Jnt1true=U0true*V1true';
Jnt2true=U0true*V2true';
Ind1true=U1true*A1true';
Ind2true=U2true*A2true';
Theta1true=Mean1true+Jnt1true+Ind1true;
Theta2true=Mean2true+Jnt2true+Ind2true;
svds(Theta1true) % get a sense of signal strength in the joint and individual structures
svds(Theta2true)


% visualize meaningful parameters (binomial: pi, poisson: lambda, normal: mean)
switch distr1
    case 'binomial'
        param1= exp(Theta1true)./(1+exp(Theta1true));
        figure(1);clf;hist(param1(:));title(['Binomial Mean Parameter'])
    case 'poisson'
        param1= exp(Theta1true);
        figure(1);clf;mesh(param1);title('Poisson Random Variables')
    case 'normal'
        param1= Theta1true;
        figure(1);clf;hist(param1(:));title(['Normal Mean Parameter'])
end;
switch distr2
    case 'binomial'
        param2= exp(Theta2true)./(1+exp(Theta2true));
        figure(2);clf;hist(param2(:));title(['Binomial Mean Parameter'])
    case 'poisson'
        param2= exp(Theta2true);
        figure(2);clf;mesh(param2);title('Poisson Random Variables')
    case 'normal'
        param2= Theta2true;
        figure(2);clf;hist(param2(:));title(['Normal Mean Parameter'])
end;

