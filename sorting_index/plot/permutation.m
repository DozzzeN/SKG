clear; close all; clc

load data_NLOS.mat

rssa = A(1:6500,1);
rssLen = length(rssa);

textSize = 16;
linewid = 1;
figure(1)
plot(1:rssLen, rssa, 'r', 'MarkerFaceColor', 'r', 'LineWidth', linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 rssLen-1])
xticklabels([0:1000:rssLen])
xlabel('Index', 'FontSize', 15, 'FontWeight', 'bold')
ylabel('RSS (dB)', 'FontSize', 15, 'FontWeight', 'bold')
print('entropy1','-depsc')

load data_NLOS_permh.mat

rssaperm = A.';
rssapermLen = length(rssaperm);

figure(2)
plot(1:rssapermLen, rssaperm, 'r', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 rssLen-1])
xticklabels([0:1000:rssLen])
xlabel('Index', 'FontSize', 15, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',15, 'FontWeight','bold')
print('entropy2','-depsc')

load data_NLOS_permf.mat

rssaperm = A.';
rssapermLen = length(rssaperm);

figure(3)
plot(1:rssapermLen, rssaperm, 'r', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 rssLen-1])
xticklabels([0:1000:rssLen])
xlabel('Index', 'FontSize', 15, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',15, 'FontWeight','bold')
print('entropy3','-depsc')