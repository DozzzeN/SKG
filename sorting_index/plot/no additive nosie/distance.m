clear; close all; clc
textSize = 16;
linewid = 1;

% Actual
% Eavesdropping IS 32 64 128 256
% Inference IS 32 64 128 256
% Imitation IS 32 64 128 256
% Stalking IS 32 64 128 256

epiImpBMRis = [0.55668605 1.05196221 1.73660714 3.22890625;
    10.578488372093023 21.215843023255815 42.304315476190474 86.965625;
    10.714752906976743 21.264898255813954 42.93675595238095 83.867578125;
    10.515988372093023 21.45675872093023 42.96391369047619 85.58203125;
    10.615915697674419 21.205668604651162 43.21949404761905 85.268359375];
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
ylim([0 165])
legend('Actual Generated Key', 'Eavesdropping Attack','Inference Attack', 'Imitation Attack', 'Stalking Attack', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Number of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('Manhattan Distance', 'FontSize',15, 'FontWeight','bold')
grid on

% fsize = 8;
% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),strcat(num2str(epiImpBMRis(i,1), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)-0.1,epiImpBMRis(i,2),strcat(num2str(epiImpBMRis(i,2), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)+0.1,epiImpBMRis(i,3),strcat(num2str(epiImpBMRis(i,3), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)+0.3,epiImpBMRis(i,4),strcat(num2str(epiImpBMRis(i,4), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
% end

print('dists','-depsc')
