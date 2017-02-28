function [out]=hard_thres(a,lam)
% a is a matrix or vector or scalar
% lam is a scalar threshold
% perform softthresholding 

ind=(abs(a)<lam);
a(ind)=0;
out=a;
