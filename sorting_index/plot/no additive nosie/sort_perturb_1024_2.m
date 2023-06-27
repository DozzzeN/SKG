clear; close all; clc
textSize = 16;
linewid = 1;

% ISm-nothing 4 5 6 7
% ISm-without sorting 4 5 6 7
% ISm-without perturbation 4 5 6 7
% ISm-RSS 4 5 6 7

% without correction
epiImpBMR = [
    1-0.6439410666 1-0.6597981771 1-0.6721940104 1-0.6842172476;
    1-0.9943529212 1-0.9996365017 1-0.9999414063 0;
    1-0.6891431726 1-0.7210883247 1-0.7458854167 1-0.7596980168;
    1-0.9995117188 0 0 0
    ];
epi = [4 5 6 7];

figure(111)
bar(epi, epiImpBMR(1:4,:)', 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 0.7]);
legend('SIM-SKG w.o. Sorting and Perturbation', 'SIM-SKG w.o. Sorting', 'SIM-SKG w.o. Perturbation', 'SIM-SKG', 'FontSize',textSize-2, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
grid on
for i=1:length(epiImpBMR)
    text(epi(i)-0.275,epiImpBMR(1,i),strcat(num2str(epiImpBMR(1,i), '%.2g')), 'FontSize',11, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)-0.105,epiImpBMR(2,i),strcat(num2str(epiImpBMR(2,i), '%.2g')), 'FontSize',11, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.105,epiImpBMR(3,i),strcat(num2str(epiImpBMR(3,i), '%.2g')), 'FontSize',11, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.275,epiImpBMR(4,i),strcat(num2str(epiImpBMR(4,i), '%.2g')), 'FontSize',11, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('epiCSIwithout2','-depsc')