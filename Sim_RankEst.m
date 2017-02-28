% This file conducts simulation studies on rank estimation
%
% Need to call:
%    Sim_Setting
%
% 11/16/2016 by Gen Li


%% load simulation settings

% for choosesetting=1:4
choosesetting=1;
sparsity=0;
Sim_Setting


%% generate data
rng(20161116) 
switch distr1
    case 'binomial'
        X1=binornd(1,param1);
    case 'poisson'
        X1=poissrnd(param1);
    case 'normal'
        X1=normrnd(param1,1); % CAN CHANGE noise if needed
end;
switch distr2
    case 'binomial'
        X2=binornd(1,param2);
    case 'poisson'
        X2=poissrnd(param2);
    case 'normal'
        X2=normrnd(param2,1);
end;
disp('Data generated!')



%% estimate individual ranks
Nfold=10;
rcand=1:6;

if strcmp(distr1,'binomial')
    [avgCVscore1,r1_est,allCVscore1]=Nfold_CV_Single(X1,distr1,rcand,Nfold,struct('lambda',0.05)); 
else
    [avgCVscore1,r1_est,allCVscore1]=Nfold_CV_Single(X1,distr1,rcand,Nfold);
end;
figure(1);clf;
subplot(1,3,1)
plot(rcand,allCVscore1','k*--','linewidth',1);
hold on;
plot(rcand,avgCVscore1,'ro-','linewidth',2); % median
set(gca,'fontsize',12,'xtick',rcand);
xlim([min(rcand),max(rcand)]);
xlabel('Rank','fontsize',15);
ylabel('CV Score','fontsize',15);
title(['X_1: ',distr1],'fontsize',15);


if strcmp(distr2,'binomial')
    [avgCVscore2,r2_est,allCVscore2]=Nfold_CV_Single(X2,distr2,rcand,Nfold,struct('lambda',0.05));
else
    [avgCVscore2,r2_est,allCVscore2]=Nfold_CV_Single(X2,distr2,rcand,Nfold);
end;
figure(1);
subplot(1,3,2)
plot(rcand,allCVscore2','k*--','linewidth',1);
hold on;
plot(rcand,avgCVscore2,'ro-','linewidth',2);
set(gca,'fontsize',12,'xtick',rcand);
xlim([min(rcand),max(rcand)]);
xlabel('Rank','fontsize',15);
ylabel('CV Score','fontsize',15);
title(['X_2: ',distr2],'fontsize',15);



%% estimate the joint rank 
Nfold=10;
rcand=max(r1_est,r2_est):(r1_est+r2_est);

if strcmp(distr1,'binomial') || strcmp(distr2,'binomial')
    [avgCVscore0,r0_est,allCVscore0]=Nfold_CV_Mixed(X1,X2,distr1,distr2,rcand,Nfold,...
        struct('lambda1',0.01,'lambda2',0.01));
else
    [avgCVscore0,r0_est,allCVscore0]=Nfold_CV_Mixed(X1,X2,distr1,distr2,rcand,Nfold);
end;
figure(1);
subplot(1,3,3)
plot(rcand,allCVscore0','k*--','linewidth',1);
hold on;
plot(rcand,median(allCVscore0,1),'ro-','linewidth',2);
set(gca,'fontsize',12,'xtick',rcand);
xlim([min(rcand),max(rcand)]);
% ylim([400,1000]);
xlabel('Rank','fontsize',15);
ylabel('CV Score','fontsize',15);
title('[X_1,X_2]: Combined','fontsize',15);
orient landscape
print('-dpdf',[ppth,'Sim',num2str(choosesetting),'_Rank_paper']);


% end;


