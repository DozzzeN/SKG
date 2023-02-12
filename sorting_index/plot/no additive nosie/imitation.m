clear; close all; clc
textSize = 16;
linewid = 1;


% Attack: Stalking
% MI IS 5 6 7 8
% SI IS 5 6 7 8
% MO IS 5 6 7 8
% SO IS 5 6 7 8

epiImpBMRis = [1-0.4960611979 1-0.5015869141 1-0.4965820312 1-0.5014105903;
    1-0.4997273235 1-0.4989089966 1-0.4999911222 1-0.5010274251;
    1-0.4948730469 1-0.5 1-0.4973958333 1-0.4923502604;
    1-0.4989947151 1-0.5042550223 1-0.4986979167 1-0.4992675781];
kle = [5, 6, 7, 8];

figure(31)
bar(kle, epiImpBMRis, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0 0.9])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR (%)', 'FontSize',15, 'FontWeight','bold')
grid on

% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)-0.1,epiImpBMRis(i,2),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.1,epiImpBMRis(i,3),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.3,epiImpBMRis(i,4),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

print('imitBMR','-depsc')
