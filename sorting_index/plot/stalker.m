clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Attack: Stalking

epiImpBMRdp = [1-0.49122807 1-0.495614035 1-0.471491228 1-0.493421053];
epiImpKMRdp = [1 1 1 1];
epiImpBMRed = [1-0.629107981 1-0.642585551 1-0.565326633 1-0.471600688];
epiImpKMRed = [1-0.298245614 1-0.214285714 1-0.071428571 1];
kle = [16, 32, 64, 128];

figure(31)
plot(kle, epiImpBMRdp, 'd-.', 'LineWidth',2)
hold on
plot(kle, epiImpBMRed, 's-', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0 0.6])
legend('BM-SKG', 'SA-SKG', 'FontSize', textSize, 'FontWeight','bold', 'Location','southeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
grid on

% for i=1:length(epiImpBMRdp)
%     text(kle(i),epiImpBMRdp(i),num2str(epiImpBMRdp(i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i),epiImpBMRed(i),num2str(epiImpBMRed(i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end
print('attackBMR','-depsc')

figure(32)
plot(kle, epiImpKMRdp, 'd-.', 'LineWidth',2)
hold on
plot(kle, epiImpKMRed, 's-', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0.6 1.05])
legend('BM-SKG', 'SA-SKG', 'FontSize', textSize, 'FontWeight','bold', 'Location','southeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
grid on

% for i=1:length(epiImpKMRdp)
%     text(kle(i),epiImpKMRdp(i),num2str(epiImpKMRdp(i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i),epiImpKMRed(i),num2str(epiImpKMRed(i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end
print('attackKMR','-depsc')