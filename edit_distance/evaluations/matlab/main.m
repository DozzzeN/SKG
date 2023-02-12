clear; close all; clc

N = 500 ; % number of steps
X = 1 ;  % number of dimensions
% positions, starting at (0,0,...,0)
P = cumsum(full(sparse(1:N, randi(X,1,N), [0 2*randi([0 1],1,N-1)-1], N, X))) ; 
% visualisation
figure ;
hold on ;
for k=1:size(P,2)
   plot(1:size(P,1),P(:,k),'.-') ;
   text(size(P,1),P(end, k), sprintf(' dim %d',k)) ;
end
xlabel('Step') ;
ylabel('Position') ;
hold off ;