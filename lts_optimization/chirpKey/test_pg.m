clear
load('A0h-gau.mat')

N=40;                                 %length of x
M=20;                                 %rows of A
K=5;                                  %support size
sA=.01/M;                             %A noise variance
sb=sA;                                %b noise variance
lam=.02;                              %regularization parameter
ni=150;                               %no. of iterations

m22=zeros(ni,1);                      %mean-square error

% rng(1);
A=A0  +sqrt(sA)*randn(M,N);       %noisy A matrix
% rng(2);
b=A0*h+sqrt(sb)*randn(M,1);       %noisy b vector

save("A.mat", "A")
save("b.mat", "b")

[e23,~,~,x]=ass_pg_stls_f(A,b,N,K,lam,h,ni);

m22=m22+e23
