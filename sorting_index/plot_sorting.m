load sorting.mat

% figure(1)
% plot(A(300:400,1))
% histogram(A(:,1))
% cdfplot(A(:,1))

% figure(2)
% plot(A(300:400,2))
% histogram(A(:,2))
% cdfplot(A(:,2))

corr(A(1:300,1), A(1:300,2),'type','pearson')