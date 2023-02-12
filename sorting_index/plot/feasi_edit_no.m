clear all; close all; clc

load data_mobile_indoor_2.mat

% Original RSS Sequence
startInd = 134+4;
len = 10;
ind = startInd:(startInd + len - 1);
rssa = A(ind,1);
rssb = A(ind,2);

textSize = 20;
linewid = 3;

figure(1)
plot(0:length(ind)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(ind)-1, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(ind)-1])
ylim([-76 -63])
legend('Alice', 'Bob', 'FontSize', textSize, 'FontWeight', 'bold', 'Location', 'northeast')
xlabel('Index', 'FontSize', 20, 'FontWeight', 'bold')
ylabel('RSS (dB)', 'FontSize', 20, 'FontWeight', 'bold')
print('editOri','-depsc')

% rperm = randperm(length(rssa));
ins = [2,4,7];
insv = [-71, -68, -73];
del = [1,8,9];
redit = zeros(1, length(rssa));
for i = 1:length(rssa)
    redit(i) = rssa(i);
end
for i = 1:length(ins)
    for j = length(redit):-1:ins(i)
        redit(j+1) = redit(j);
    end
    redit(ins(i)) = insv(i);
end
for i = 1:length(del)
    for j = del(i):length(redit)-1
        redit(j) = redit(j+1);
    end
end
redit =  redit(1:length(rssa)+length(ins)-length(del));

figure(11)
plot(0:length(ind)-1, redit, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(ind)-1, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(ind)-1])
ylim([-76 -63])
legend('Alice', 'Bob', 'FontSize', textSize, 'FontWeight', 'bold', 'Location', 'northeast')
xlabel('Index', 'FontSize', 20, 'FontWeight', 'bold')
ylabel('RSS (dB)', 'FontSize', 20, 'FontWeight','bold')
print('editEdit','-depsc')

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

figure1 = figure(2);
plot(0:length(ind)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(ind)-1, rssb(PM(minp,:)), 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(ind)-1])
ylim([-76 -63])
legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Index', 'FontSize', 20, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize', 20, 'FontWeight','bold')

print('editMatch','-depsc')