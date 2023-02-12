clear; close all; clc
textSize = 16;
linewid = 1;

% -----------------------------------
% Impact of Entropy-based permutation
% w. perm
% w.o. perm

permImpBMR = [1-0.999668435 0 0 0; 1-0.989051095 1-0.961928934 0 1-0.998668442];
permImpKMR = [1-0.993630573 0 0 0; 1-0.974522293 1-0.966666667 0 1-0.971428571];

figure(12)
bar([1 2 3 4], permImpBMR, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

ylim([0 0.05])
legend('w. Entropy Permutation','w.o. Entropy Permutation', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('permImpBMR','-depsc')


figure(13)
bar([1 2 3 4], permImpKMR, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

ylim([0 0.05])
legend('w. Entropy Permutation','w.o. Entropy Permutation', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('permImpKMR','-depsc')