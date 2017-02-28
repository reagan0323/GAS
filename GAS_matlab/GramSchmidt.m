function [Q1,Q,R] = GramSchmidt(A1,B)
% A1 = randn(p1+p2,r0);
% compute QR of A1 using Gram-Schmidt, orthogonal to (orthogonal) B.

[m,n] = size(A1);

if nargin == 2
    [m1,n2] = size(B);
    
    if m ~=m1; return;
    end
    
    A=[B A1];
    
    for j = 1:n+n2
        v = A(:,j);
        for i=1:j-1
            R(i,j) = Q(:,i)'*A(:,j);
            v = v - R(i,j)*Q(:,i);
        end
        R(j,j) = norm(v);
        Q(:,j) = v/R(j,j);
    end
    
    Q1= Q(:,n2+1:end);
    
    
else
    A = A1;
    
    for j = 1:n
        v = A(:,j);
        for i=1:j-1
            R(i,j) = Q(:,i)'*A(:,j);
            v = v - R(i,j)*Q(:,i);
        end
        R(j,j) = norm(v);
        Q(:,j) = v/R(j,j);
    end
    
    Q1= Q;
    
end