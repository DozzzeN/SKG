clear all; close all; clc

load data_mobile_indoor_2.mat

% Original RSS Sequence
ind = 116:143;
rssa = A(ind,1);
rssb = A(ind,2);

figure(1)
% subplot(211)
plot(0:27, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:27, ones(1,28)*-65, 'k-', 'LineWidth',2)
xlim([-1 28])
ylim([-80 -50])
% subplot(212)
figure(11)
plot(0:27, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(0:27, ones(1,28)*-65, 'k-', 'LineWidth',2)
xlim([-1 28])
ylim([-80 -50])

% 1st Permuted RSS Sequence
textSize = 35;
perm = randperm(28);
permInd = ind(perm);
rssa = A(permInd,1);
rssb = A(permInd,2);

figure(2)
% subplot(211)
plot(0:27, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(ones(1,4)*-0.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*6.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*13.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*20.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*27.1, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*34.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^A_' num2str(p)];
    text((p-1)*7+1, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end

xlim([-1 28])
ylim([-85 -50])

% subplot(212)
figure(21)
plot(0:27, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*-0.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*6.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*13.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*20.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*27.1, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*34.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^B_' num2str(p)];
    text((p-1)*7+1, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([-1 28])
ylim([-85 -50])

% 2nd Permuted RSS Sequence
permEpi = [2 4 1 3];
permEpiInd = [];

for p = 1:4
    permEpiInd = [permEpiInd, ((permEpi(p)-1)*7+1):permEpi(p)*7];
end
permInd = permInd(permEpiInd);

rssa = A(permInd,1);
% rssb = A(permInd,2);

figure(3)
% subplot(211)
plot(0:27, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(ones(1,4)*-0.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*6.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*13.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*20.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*27.1, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*34.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^A_' num2str(permEpi(p))];
    text((p-1)*7+1, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([-1 28])
ylim([-85 -50])

% subplot(212)
figure(31)
plot(0:27, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*-0.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*6.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*13.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*20.5, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,4)*27.1, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*34.1, -80:10:-50, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^B_' num2str(p)];
    text((p-1)*7+1, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([-1 28])
ylim([-85 -50])




