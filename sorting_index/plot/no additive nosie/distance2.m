clear; close all; clc
textSize = 16;
linewid = 1;

% Actual
% Eavesdropping IS 32 64 128 256
% Inference IS 32 64 128 256
% Imitation IS 32 64 128 256
% Stalking IS 32 64 128 256

% epiImpBMRis = [0.5912064 0.98728198 1.87202381 2.809375;
%     10.51453488 21.30741279 42.11011905 86.02578125;
%     10.59920058 21.5090843 42.43303571 86.08359375;
%     10.64244186 22.01744186 42.37276786 86.23789063;
%     10.40116279 20.93023256 42.56436012 85.5953125];
epiImpBMRis = [10.51453488 21.30741279 42.11011905 86.02578125;
    10.59920058 21.5090843 42.43303571 86.08359375;
    10.64244186 22.01744186 42.37276786 86.23789063;
    10.40116279 20.93023256 42.56436012 85.5953125];
kle = [5, 6, 7, 8];

epiImpBMRis = epiImpBMRis';

figure(31)
bar(kle, epiImpBMRis, 'grouped')
hold on
xticklabels({'32','64','128','256'})
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 165])
legend('Random Guess Attack','Inference Attack', 'Imitation Attack', 'Stalking Attack', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Number of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('Average Difference', 'FontSize',15, 'FontWeight','bold')
grid on

% fsize = 8;
% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),strcat(num2str(epiImpBMRis(i,1), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)-0.1,epiImpBMRis(i,2),strcat(num2str(epiImpBMRis(i,2), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)+0.1,epiImpBMRis(i,3),strcat(num2str(epiImpBMRis(i,3), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)+0.3,epiImpBMRis(i,4),strcat(num2str(epiImpBMRis(i,4), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
% end

print('dists','-depsc')
