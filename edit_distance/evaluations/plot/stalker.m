clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Attack: Stalking

epiImpBMR = [1-0.51048951 1-0.410071942 1-0.238095238 1-0.1015625];
epiImpKMR = [1-0.431034483 1-0.137931034 1-0.071428571 1];
kle = [16, 32, 64, 128];

figure(31)
plot(kle, epiImpBMR, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0.4 1])
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('attackBMR','-depsc')

figure(32)
plot(kle, epiImpKMR, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0.5 1])
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('attackKMR','-depsc')