clear all; close all; clc

textSize = 35;
noteSize = 23;

% Original RSS Sequence
rssa = [-52,-54,-54,-48,-44,-43,-51,-50,-51,-51,-49,-51];
rssb = [-54,-53,-54,-48,-44,-43,-52,-52,-51,-51,-51,-50];

rssa = rssa - min(rssa)+1;
rssb = rssb - min(rssb)+1;
xliml = -1;
xlimr = 12;
ylimd = -5;
ylimu = 15;
range = ylimd:3:ylimu;

figure1=figure(1);
bar(0:length(rssa)-1, rssa, 'r', 'grouped')
hold on
ori_ind = [1,2,3,4,5,6,7,8,9,10,11,12];
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/4*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)), range, 'b--', 'LineWidth', 2)
hold on
for i=1:length(ori_ind)
    text((i-1),14,strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),rssa(i),strcat(num2str(rssa(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
    hold on
end
for p = 1:4
    txt = ['s^A_' num2str(p)];
    text((p-1)*(length(rssa)/4)+0.25, ylimd+2.5, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu+2])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 rectangle
annotation(figure1,'rectangle',...
    [0.517857142857143 0.103968253968254 0.178571428571428 0.823015873015873],...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
    'LineWidth',3);
print('ind_a','-depsc')

figure2=figure(2);
bar(0:length(rssb)-1, rssb, 'k', 'grouped')
hold on
ori_ind = [1,2,3,4,5,6,7,8,9,10,11,12];
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/4*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)), range, 'b--', 'LineWidth', 2)
hold on
for i=1:length(ori_ind)
    text((i-1),14,strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),rssb(i),strcat(num2str(rssb(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
    hold on
end
for p = 1:4
    txt = ['s^B_' num2str(p)];
    text((p-1)*(length(rssa)/4)+0.25, ylimd+2.5, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu+2])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 rectangle
annotation(figure2,'rectangle',...
    [0.696238095238095 0.103174603174604 0.179952380952381 0.822222222222222],...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
    'LineWidth',3);
print('ind_b','-depsc')

% Index
inda=[2, 3, 1, 7, 9, 10, 12, 8, 11, 4, 5, 6];
indb=[1, 3, 2, 7, 8, 9, 10, 11, 12, 4, 5, 6];

% sorting
figure3=figure(3);
rssa=sort(rssa);
bar(0:length(rssa)-1, rssa, 'r', 'grouped')
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/4*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssa)), range, 'b--', 'LineWidth', 2)
hold on
for i=1:length(ori_ind)
    text((i-1),rssa(i),strcat(num2str(rssa(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
    text((i-1),14,strcat(num2str(inda(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
for p = 1:4
    txt = ['e^A_' num2str(p)];
    text((p-1)*(length(rssa)/4)+0.25, ylimd+2.5, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu+2])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 rectangle
annotation(figure3,'rectangle',...
    [0.339690476190475 0.104761904761905 0.176380952380952 0.817460317460318],...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
    'LineWidth',3);

print('sort_ind_a','-depsc')


figure4=figure(4);
rssb=sort(rssb);
bar(0:length(rssb)-1, rssb, 'k', 'grouped')
hold on
hold on
plot(ones(1,length(range))*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/4*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(-0.5+length(rssb)), range, 'b--', 'LineWidth', 2)
hold on
for i=1:length(ori_ind)
    text((i-1),rssb(i),strcat(num2str(rssb(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontAngle','italic');
    text((i-1),14,strcat(num2str(indb(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
for p = 1:4
    txt = ['e^B_' num2str(p)];
    text((p-1)*(length(rssb)/4)+0.25, ylimd+2.5, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu+2])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 rectangle
annotation(figure4,'rectangle',...
    [0.518857142857143 0.103174603174603 0.178761904761905 0.817460317460318],...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
    'LineWidth',3);
print('sort_ind_b','-depsc')
