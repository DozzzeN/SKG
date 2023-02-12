clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------

% -----------------------------------
% time complexity
epiImp = [ 0.01 0.052  0.24 1.7  ];
kle = [16, 32, 64, 128]*7;

figure(13)
plot(kle, epiImp, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
% ylim([-76 -63])
% legend('NLOS', 'LOS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Key length (bit)', 'FontSize', 15, 'FontWeight','bold')
ylabel('Time (sec.)', 'FontSize',15, 'FontWeight','bold')
print('timeCmplx','-depsc')

% -----------------------------------
% Impact of CSI
epiImp = [4.37e-4, 0 0 0; 3.59e-4, 0 0 0];
epi = [7, 8, 9, 10];

figure(111)
bar(epi, epiImp(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
ylim([0 1.5e-3])
legend('CSI','RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiCSIBMR','-depsc')

epiImp = [3*6/7 3*6/8 3*6/9 3*6/10; 6/7 6/8 6/9 6/10];
figure(112)
bar(epi, epiImp(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
% ylim([0 1.5e-3])
legend('CSI','RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BGR', 'FontSize',15, 'FontWeight','bold')
print('epiCSIBGR','-depsc')

% -----------------------------------
% Attack: Stalking

% 16: 0.387   32: 0.469   64: 0.494 128: 129: 50.6
epiImp = [ 0.387 0.469  0.494  0.506  ];
kle = [16, 32, 64, 128];

figure(3)
plot(kle, epiImp, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
% ylim([-76 -63])
% legend('NLOS', 'LOS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('attackBMR','-depsc')

% --------------------------------------
% Attack: predicatable channel
load devicefree25.mat 
rss = carrierAmp25-80;
len = length(rss);

figure(4)
plot(rss, 'r-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 len])
% ylim([-76 -63])
% legend('NLOS', 'LOS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Index', 'FontSize', 15, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',15, 'FontWeight','bold')
print('attack1','-depsc')

pind = randperm(length(rss));
prss = rss(pind);

figure(41)
plot(prss, 'r-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 len])
% ylim([-76 -63])
% legend('NLOS', 'LOS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Index', 'FontSize', 15, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',15, 'FontWeight','bold')
print('attack2','-depsc')




