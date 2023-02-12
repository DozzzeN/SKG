clear all; close all; clc

load data_mobile_indoor_2.mat

% Original RSS Sequence
startInd = 134;
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
legend('Alice', 'Bob', 'FontSize', textSize, 'FontWeight', 'bold', 'Location', 'southeast')
xlabel('Index', 'FontSize', 20, 'FontWeight', 'bold')
ylabel('RSS (dB)', 'FontSize', 20, 'FontWeight', 'bold')
print('editOri','-depsc')

% rperm = randperm(length(rssa));
ins = [4,7];
insv = [-68, -73];
del = [];
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

figure11 =  figure(11);
plot(0:length(redit)-1, redit, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
plot(0:length(rssb)-1, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(redit)-1])
ylim([-76 -63])
legend('Alice', 'Bob', 'FontSize', textSize, 'FontWeight', 'bold', 'Location', 'southeast')
xlabel('Index', 'FontSize', 20, 'FontWeight', 'bold')
ylabel('RSS (dB)', 'FontSize', 20, 'FontWeight','bold')

% 创建 ellipse
annotation(figure11,'ellipse',...
    [0.535523809523809 0.255555555555557 0.0537619047619048 0.196825396825398],...
    'Color',[0 0 1],...
    'LineWidth',3,...
    'LineStyle','-.');

% 创建 ellipse
annotation(figure11,'ellipse',...
    [0.328380952380952 0.531746031746032 0.0537619047619048 0.196825396825398],...
    'Color',[0 0 1],...
    'LineWidth',3,...
    'LineStyle','-.');

% 创建 arrow
annotation(figure11,'arrow',[0.472619047619048 0.559523809523809],...
    [0.80952380952381 0.473015873015873],'Color',[0 0 1],'LineWidth',3);

% 创建 arrow
annotation(figure11,'arrow',[0.395238095238095 0.36547619047619],...
    [0.803174603174603 0.746031746031746],'Color',[0 0 1],'LineWidth',3);

% 创建 textbox
annotation(figure11,'textbox',...
    [0.182142857142857 0.81646031746032 0.603571428571429 0.103174603174603],...
    'Color',[0 0 1],...
    'String',{'Inserted redundant RSS'},...
    'FontWeight','bold',...
    'FontSize',20,...
    'FitBoxToText','off',...
    'EdgeColor','none');
print('editEdit','-depsc')

% ori = [1 3 5 6 8 9 10 11 12 13];
% rssbe = redit;
% j=1;
% for i = 1:length(ori)
%     rssbe(ori(i)) = rssb(j);
%     j = j+1;
% end

rssbe1 = rssb([1 2 3]);
rssbe2 = rssb([4 5]);
rssbe3 = rssb([6 7 8 9 10]);

figure2 = figure(2);
plot(0:length(redit)-1, redit, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',linewid)
hold on
% plot([0 1 2], rssbe1, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
% plot([3 5], rssbe2, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
% plot([7 8 9 10 11], rssbe3, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
plot([0 1 2 3 5 7 8 9 10 11], rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',linewid)
hold on
ax = gca;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 length(redit)-1])
ylim([-76 -63])
legend('Alice','Bob', 'FontSize',textSize, 'FontWeight','bold', 'Location','southeast')
xlabel('Index', 'FontSize', 20, 'FontWeight','bold')
ylabel('RSS (dB)', 'FontSize', 20, 'FontWeight','bold')

% 创建 textbox
annotation(figure2,'textbox',...
    [0.217857142857143 0.799000000000001 0.447619047619048 0.103174603174603],...
    'Color',[0 0 1],...
    'String',{'Mismatched RSS'},...
    'FontWeight','bold',...
    'FontSize',20,...
    'EdgeColor','none');

% 创建 ellipse
annotation(figure2,'ellipse',...
    [0.326 0.538095238095238 0.0573333333333334 0.185714285714286],...
    'Color',[0 0 1],...
    'LineWidth',3,...
    'LineStyle','-.');

% 创建 arrow
annotation(figure2,'arrow',[0.397619047619048 0.369047619047619],...
    [0.796825396825397 0.741269841269842],'Color',[0 0 1],'LineWidth',3);

print('editMatch','-depsc')