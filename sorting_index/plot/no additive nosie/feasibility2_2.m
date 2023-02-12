clear all; close all; clc

textSize = 35;
noteSize = 23;

% Original RSS Sequence
rssa = [-52, -54, -53, -48, -41, -48, -47, -45, -47.5];
rssb = [-51.5, -53.5, -52.5, -49, -50, -48, -46.5, -44.5, -47];

% rssa = rssa - min(rssa)+1;
% rssb = rssb - min(rssb)+1;
xliml = -1;
xlimr = 9;
ylimd = -63;
ylimu = -30;
range = ylimd:3:ylimu;

figure1=figure(1);
plot(0:length(rssa)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:8, ones(1,9)*mean(rssa), 'k-', 'LineWidth',2)
hold on
ori_ind = [1,2,3,4,5,6,7,8,9];
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/3*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)), range, 'b--', 'LineWidth', 2)
hold on
% for i=1:length(ori_ind)
%     text((i-1),8,strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
%     text((i-1),rssa(i),strcat(num2str(rssa(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
%     hold on
% end
for p = 1:3
    txt = ['s^A_' num2str(p)];
    text((p-1)*(length(rssa)/3)+0.25, ylimd+4, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 rectangle
% annotation(figure1,'rectangle',...
%     [0.517857142857143 0.103968253968254 0.178571428571428 0.823015873015873],...
%     'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
%     'LineWidth',3);
% 创建 ellipse
annotation(figure1,'ellipse',...
    [0.468857142857143 0.523809523809527 0.0870952380952381 0.217460317460322],...
    'Color',[0.0745098039215686 0.623529411764706 1],...
    'LineWidth',4);
print('ind_a','-depsc')

figure2=figure(2);
plot(0:length(rssb)-1, rssb, 'ks-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:8, ones(1,9)*mean(rssa), 'k-', 'LineWidth',2)
hold on
ori_ind = [1,2,3,4,5,6,7,8,9];
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/3*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)), range, 'b--', 'LineWidth', 2)
hold on
% for i=1:length(ori_ind)
%     text((i-1),8,strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
%     text((i-1),rssb(i),strcat(num2str(rssb(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
%     hold on
% end
for p = 1:3
    txt = ['s^B_' num2str(p)];
    text((p-1)*(length(rssa)/3)+0.25, ylimd+4, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% % 创建 rectangle
% annotation(figure2,'rectangle',...
%     [0.696238095238095 0.103174603174604 0.179952380952381 0.822222222222222],...
%     'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
%     'LineWidth',3);

% 创建 ellipse
annotation(figure2,'ellipse',...
    [0.474809523809524 0.350000000000002 0.0799523809523808 0.188888888888892],...
    'Color',[0.0745098039215686 0.623529411764706 1],...
    'LineWidth',4);


print('ind_b','-depsc')

% Index
inda=[2, 3, 1, 4, 6, 9, 7, 8, 5];
indb=[2, 3, 1, 5, 4, 6, 9, 7, 8];

% sorting
figure3=figure(3);
rssa=sort(rssa);
plot(0:length(rssa)-1, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:8, ones(1,9)*mean(rssa), 'k-', 'LineWidth',2)
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/3*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)), range, 'b--', 'LineWidth', 2)
hold on
for i=1:length(ori_ind)
%     text((i-1),rssa(i),strcat(num2str(rssa(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
    text((i-1),-34,strcat(num2str(inda(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
for p = 1:3
    txt = ['t^A_' num2str(p)];
    text((p-1)*(length(rssa)/3)+0.5, ylimd+4, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 ellipse
annotation(figure3,'ellipse',...
    [0.789095238095237 0.530158730158732 0.0763809523809524 0.246031746031747],...
    'Color',[0.0745098039215686 0.623529411764706 1],...
    'LineWidth',4);

print('sort_ind_a','-depsc')


figure4=figure(4);
rssb=sort(rssb);
plot(0:length(rssb)-1, rssb, 'ks-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:8, ones(1,9)*mean(rssa), 'k-', 'LineWidth',2)
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/3*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)), range, 'b--', 'LineWidth', 2)
hold on
for i=1:length(ori_ind)
%     text((i-1),rssb(i),strcat(num2str(rssb(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
    text((i-1),-34,strcat(num2str(indb(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
for p = 1:3
    txt = ['t^B_' num2str(p)];
    text((p-1)*(length(rssb)/3)+0.5, ylimd+4, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 ellipse
annotation(figure4,'ellipse',...
    [0.404761904761904 0.346031746031748 0.0642857142857145 0.18730158730159],...
    'Color',[0.0745098039215686 0.623529411764706 1],...
    'LineWidth',4);
print('sort_ind_b','-depsc')
