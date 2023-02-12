clear; close all; clc

load data_NLOS.mat

% Original RSS Sequence
rssa = A(:,1);
rssb = A(:,2);
rssLen = length(rssa);

% rssa = A(1:rssLen/8,1);
% rssb = A(1:rssLen/8,2);
rssa = A((rssLen/8-500):rssLen/8,1);
rssb = A((rssLen/8-500):rssLen/8,2);
rssLen = length(rssa);


textSize = 16;
linewid = 1;
figure(1)
plot(1:rssLen, rssa, 'r', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 rssLen-1])
% ylim([-76 -63])
% legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','southeast')
xlabel('Index', 'FontSize', 15, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',15, 'FontWeight','bold')

rperm = randperm(length(rssa));

figure(2)
plot(1:rssLen, rssa(rperm), 'r', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 rssLen])
% ylim([-76 -63])
% legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','southeast')
xlabel('Index', 'FontSize', 15, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',15, 'FontWeight','bold')

