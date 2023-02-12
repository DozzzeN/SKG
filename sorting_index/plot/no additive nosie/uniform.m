clear; close all; clc
textSize = 16;
linewid = 1;

% MI
% Inference IS 1 2 3 4
% Imitation IS 1 2 3 4
% Stalking IS 1 2 3 4

epiImpBMRis = [
    1-0.5043945312,1-0.4912109375,1-0.4982910156,1-0.5021972656;
    1-0.4943847656,1-0.4948730469,1-0.5080566406,1-0.5024414062;
    1-0.4895019531,1-0.5068359375,1-0.5004882812,1-0.5068359375];
kle = [1, 2, 3, 4];

figure(31)
bar(kle, epiImpBMRis, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0 0.9])
legend('Inference Attack', 'Imitation Attack', 'Stalking Attack', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR (%)', 'FontSize',15, 'FontWeight','bold')

ax.XTickLabel={'σ','2σ','3σ','4σ'};
% set(axes1,'XTick',[1 2 3 4],'XTickLabel',{'σ','2σ','3σ','4σ'});
grid on

% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)-0.1,epiImpBMRis(i,2),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.1,epiImpBMRis(i,3),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.3,epiImpBMRis(i,4),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

print('uniformBMR','-depsc')
