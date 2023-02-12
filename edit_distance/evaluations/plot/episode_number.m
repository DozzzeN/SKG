clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Impact of episode
% mobile_indoor stationary_indoor mobile_outdoor stationary_outdoor

% kleImpBMR = [0 0 0 0; 0 4.3e-6 9.42e-6 3.77e-5; 0 0 0 0; 0 0 0 3.4e-6 ];
kleImpBMR = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 1-0.999666889 0 0];
kleImpKMR = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 1-0.993630573 0 0];
epi = [4, 5, 6, 7];
kle = [16, 32, 64, 128];

figure(2)
bar(epi, kleImpBMR(1:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'16','32','64','128'})
ylim([0 0.0006])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('kleImpBMR','-depsc')


% figure(211)
% bar(epi, kleImpBMR(1:2,:), 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% xticklabels({'16','32','64','128'})
% % ylim([0 5e-5])
% legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
% ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
% print('kleImpIndoorBMR','-depsc')

% figure(212)
% bar(epi, kleImpBMR(3:4,:), 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% xticklabels({'16','32','64','128'})
% % ylim([0 5e-5])
% legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
% ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
% print('kleImpOutdoorBMR','-depsc')


figure(3)
bar(epi, kleImpKMR(1:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'16','32','64','128'})
ylim([0 0.012])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('kleImpKMR','-depsc')


% figure(311)
% bar(epi, kleImpKMR(1:2,:), 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% xticklabels({'16','32','64','128'})
% % ylim([0 5e-5])
% legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
% ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
% print('kleImpIndoorKMR','-depsc')

% figure(312)
% bar(epi, kleImpKMR(3:4,:), 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% xticklabels({'16','32','64','128'})
% % ylim([0 5e-5])
% legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
% ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
% print('kleImpOutdoorKMR','-depsc')


% figure(213)
% bar(epi, [4, 5, 6, 7]/7, 'grouped')
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% % xlim([0 rssLen-1])
% ylim([0 1.3])
% % legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
% ylabel('BGR', 'FontSize',15, 'FontWeight','bold')
% print('kleImpKGR','-depsc')