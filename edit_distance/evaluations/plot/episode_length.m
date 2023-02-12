clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Impact of episode
% mobile_indoor stationary_indoor mobile_outdoor stationary_outdoor
epiImpBMR = [1-0.999120824 0 0 0 0; 1-0.999524715 1-0.999666889 0 0 0; 0 0 0 0 0; 1-0.996794872 0 0 0 0];
epiImpKMR = [1-0.989361702 0 0 0 0; 1-0.987261146 1-0.993630573 0 0 0; 0 0 0 0 0; 1-0.914285714 0 0 0 0];
epi = [4, 5, 6, 7, 8];

figure(1)
bar(epi, epiImpBMR(1:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold'; 
% ylim([0 5.5e-3])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpBMR','-depsc')

figure(2)
bar(epi, epiImpKMR(1:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold'; 
% ylim([0 0.15])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpKMR','-depsc')

figure(111)
bar(epi, epiImpBMR(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% ylim([0 5.5e-3])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpIndoorBMR','-depsc')

figure(112)
bar(epi, epiImpBMR(3:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% ylim([0 0.15])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpOutdoorBMR','-depsc')

figure(121)
bar(epi, epiImpKMR(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% ylim([0 5.5e-3])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpIndoorKMR','-depsc')

figure(122)
bar(epi, epiImpKMR(3:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% ylim([0 0.15])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpOutdoorKMR','-depsc')


% figure(113)
% bar(epi, 6./[5, 6, 7, 8], 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% ylim([0 1.3])
% xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
% ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
% print('epiImpKGR','-depsc')