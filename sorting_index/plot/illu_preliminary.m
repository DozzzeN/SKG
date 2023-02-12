clear all; close all; clc

load data_mobile_indoor_2.mat

% Original RSS Sequence
startInd = 105;
startInd = 133;
startInd = 134;
len = 10;
ind = startInd:(startInd + len - 1);
rssa = A(ind,1);
rssb = A(ind,2);

% rssb = rssb([1     2     4     3     5     6     7     8     9    10]);

textSize = 20;
linewid = 3;

figure(1)
plot(0:length(ind)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(ind)-1, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
% plot(0:length(ind)-1, ones(1,length(ind))*-75, 'k-', 'LineWidth',2)
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(ind)-1])
ylim([-76 -63])
legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Index', 'FontSize', 20, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',20, 'FontWeight','bold')


rperm = randperm(length(rssa));
rssa = rssa(rperm);

figure(11)
plot(0:length(ind)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(ind)-1, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
% plot(0:length(ind)-1, ones(1,length(ind))*-75, 'k-', 'LineWidth',2)
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(ind)-1])
ylim([-76 -63])
legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Index', 'FontSize', 20, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',20, 'FontWeight','bold')

% ---------------------------------------------------------------

PM = perms(1:length(ind));
[m,n] = size(PM);

minp = 1;
prevSum = 100000;
for p = 1:m
    curtSum = sum(abs(rssb(PM(p,:)) - rssa));
    if  curtSum < prevSum
        prevSum = curtSum;
        minp = p;
    end
end
disp(PM(minp,:))

figure(2)
plot(0:length(ind)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(ind)-1, rssb(PM(minp,:)), 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
% plot(0:length(ind)-1, ones(1,length(ind))*-75, 'k-', 'LineWidth',2)
ax = gca;
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(ind)-1])
ylim([-76 -63])
legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Index', 'FontSize', 16, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize',16, 'FontWeight','bold')
