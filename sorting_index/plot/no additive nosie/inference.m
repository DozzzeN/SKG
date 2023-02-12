clear; close all; clc
textSize = 16;
linewid = 1;


% Attack: Stalking
% MI IS 5 6 7 8
% SI IS 5 6 7 8
% MO IS 5 6 7 8
% SO IS 5 6 7 8

epiImpBMRis = [1-0.4970377604 1-0.5014648438 1-0.4965820312 1-0.5023328993;
    1-0.5036010742 1-0.5004185268 1-0.4952799479 1-0.498046875;
    1-0.4951171875 1-0.5020751953 1-0.4977213542 1-0.4921875;
    1-0.4982766544 1-0.5041155134 1-0.5002034505 1-0.4985351562];
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

print('inferBMR','-depsc')
