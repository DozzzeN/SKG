clear; close all; clc
textSize = 16;
linewid = 1;


% Attack: Stalking
% MI IS 5 6 7 8
% SI IS 5 6 7 8
% MO IS 5 6 7 8
% SO IS 5 6 7 8

epiImpBMRis = [1-0.5011696039 1-0.5041992187 1-0.5068847656 1-0.5117750901;
    1-0.5003051758 1-0.5129743304 1-0.5048014323 1-0.5048014323;
    1-0.5025634766 1-0.4993652344 1-0.4997558594 1-0.4938616071;
    1-0.5000878906 1-0.4996903582 1-0.4990373884 1-0.5057291667];
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

print('stalkerBMR','-depsc')
