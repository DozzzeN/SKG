clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Impact of CSI
epiImpBMR = [1-0.9932432432432432 0 0 0 0; 1-0.999524715 1-0.999666889 0 0 0];
epilmpKMR = [1-0.8333333333333334 0 0 0 0; 1-0.987261146 1-0.993630573 0 0 0];
epi = [4 5 6 7 8];

figure(111)
bar(epi, epiImpBMR(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
legend('CSI','RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiCSIBMR','-depsc')

figure(112)
bar(epi, epilmpKMR(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
legend('CSI','RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('epiCSIBMR','-depsc')

% epiImp = [3*6/7 3*6/8 3*6/9 3*6/10; 6/7 6/8 6/9 6/10];
% figure(113)
% bar(epi, epiImp(1:2,:), 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% % xlim([0 rssLen-1])
% % ylim([0 1.5e-3])
% legend('CSI','RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
% ylabel('BGR', 'FontSize',15, 'FontWeight','bold')
% print('epiCSIBGR','-depsc')
