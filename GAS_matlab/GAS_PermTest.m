function [Pval,OrigTestStat,PermTestStat]=GAS_PermTest(Theta1,Theta2,nperm)
% This function performs permutation test for significance of
% association under the generalized association study framework. It only
% needs the estimated (column-centered) Theta1 and Theta2, and is super
% fast (no model refitting needed). 
%
% Input
%         Theta1   n*p1 natural parameter matrix 
%         Theta2   n*p2 natural parameter matrix 
%         nperm     number of permutations
%
%     Output
%         Pval            1*2 p-val vectors, [parametric P, empirical P] 
%         OrigTestStat    original test statistic, rho
%         PermTestStat    permuted test statstics, a vector of length nperm
% 
% By Gen Li, 1/13/2017

[n,p1]=size(Theta1);
[n2,p2]=size(Theta2);
if n2~=n 
    error('Mismatched samples!');
end;
if max([abs(mean(Theta1,1)),abs(mean(Theta2,1))])>1E-6
    disp('Natural parameter matrix NOT centered. Column centering...');
    Theta1=bsxfun(@minus,Theta1,mean(Theta1,1));
    Theta2=bsxfun(@minus,Theta2,mean(Theta2,1));
end;


% association coefficient
denominator= norm(Theta1,'fro')*norm(Theta2,'fro');
OrigTestStat=sum(svd(Theta1'*Theta2,'econ'))/denominator;

%% Permutation Test
PermTestStat=zeros(nperm,1);
for iperm=1:nperm
    rng('shuffle');
    ind=randperm(n);
    newTheta2=Theta2(ind,:);
    PermTestStat(iperm)=sum(svd(Theta1'*newTheta2,'econ'))/denominator;
end;



%% One-Sided Test
empPval=sum(PermTestStat>OrigTestStat)/nperm; % empirical p value
thrPval=1-normcdf(OrigTestStat,mean(PermTestStat),std(PermTestStat)); % normal theoretical p value
Pval=[thrPval,empPval];

figure();clf;
plot(PermTestStat,rand(length(PermTestStat),1)+1,'o');
hold on;
plot(OrigTestStat,1.5,'rX');
ylim([0,3]);
title(['P Value: ',num2str(Pval(1)),'(theoretical), ',num2str(Pval(2)),'(empirical)' ]);

           