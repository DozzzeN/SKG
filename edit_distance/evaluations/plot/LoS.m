clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Impact of LoS (both indoor Stationary)
% NLoS
% LoS
epiImpBMR = [1-0.999085923 0 0 0; 1-0.999697794 0 0 0];
epiImpKMR = [1-0.985294118 0 0 0; 1-0.992647059 0 0 0];
epi = [4, 5, 6, 7];

figure(13)
bar(epi, epiImpBMR, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

ylim([0 0.0011])
legend('NLoS', 'LoS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('losImpBMR','-depsc')

figure(14)
bar(epi, epiImpKMR, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

ylim([0 0.016])
legend('NLoS', 'LoS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('losImpKMR','-depsc')