clear; close all; clc
textSize = 16;
linewid = 1;


% Eavesdropping IS 5 6 7 8
% Inference IS 5 6 7 8
% Imitation IS 5 6 7 8
% Stalking IS 5 6 7 8

epiImpBMRis = [1-0.4952473958 1-0.4972330729 1-0.4909179687 1-0.4950086806;
    1-0.4970377604 1-0.5014648438 1-0.4965820312 1-0.5023328993;
    1-0.4960611979 1-0.5015869141 1-0.4965820312 1-0.5014105903;
    1-0.5011696039 1-0.5041992187 1-0.5068847656 1-0.5117750901];
kle = [5, 6, 7, 8];

figure(31)
bar(kle, epiImpBMRis, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 0.9])
yticks(0:0.1:0.9)
legend('Random Guess Attack','Inference Attack', 'Imitation Attack', 'Stalking Attack', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
grid on

% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)-0.1,epiImpBMRis(i,2),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.1,epiImpBMRis(i,3),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.3,epiImpBMRis(i,4),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

print('attacksBMR','-depsc')
