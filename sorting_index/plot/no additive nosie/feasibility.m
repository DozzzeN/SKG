clear all; close all; clc

textSize = 35;
noteSize = 25;

% Original RSS Sequence
% rssa = [-63, -64, -60, -62, -65, -62, -61, -63, -64, -63, -66, -64, -64, -62, -63, -62, -61, -63, -64, -62];
% rssb = [-62, -63, -60, -62, -65, -62, -61, -63, -64, -62, -66, -64, -63, -63, -64, -62, -61, -63, -64, -63];
rssa = [-68.5, -58, -67.5, -62.5, -60, -61, -64, -59, -66, -67, -65.5, -66.5, -60.5, -61.5, -63.5, -59.5];
rssb = [-67.5, -58.5, -68.5, -62.5, -60, -61, -63, -59, -66, -67, -65, -66.5, -60.5, -61.5, -64.5, -59.5];
xliml = -1;
xlimr = 16;
ylimd = -73;
ylimu = -55;
range = ylimd:2:ylimu;

figure1=figure(1);
plot(0:15, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:15, ones(1,16)*mean(rssa), 'k-', 'LineWidth',2)
hold on
plot(ones(1,length(range))*0, range, 'b--', 'LineWidth', 2)
hold on
% plot(ones(1,length(range))*(15/4), range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,length(range))*(15/4*2), range, 'b--', 'LineWidth', 2)
% hold on
plot(ones(1,length(range))*(15/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*15, range, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['s^A_' num2str(p)];
    text((p-1)*(15/4)+(15/4)/4, ylimd+3, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% 创建 rectangle
annotation(figure1,'rectangle',...
    [0.342857142857142 0.114285714285714 0.174404761904762 0.814285714285715],...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
    'LineWidth',2);

% % 创建 rectangle
% annotation(figure1,'rectangle',...
%     [0.692261904761903 0.107142857142857 0.174404761904762 0.814285714285715],...
%     'Color',[0.392156862745098 0.831372549019608 0.0745098039215686],...
%     'LineWidth',2,...
%     'LineStyle','--');

% annotation(figure1,'textbox',...
%     [0.335523809523808 0.790476190476198 0.159714285714286 0.126984126984128],...
%     'String',{'Incorrectly','Matched','with $s^B_4$'},...
%     'Interpreter','latex',...
%     'FontWeight','bold',...
%     'FontSize',15,...
%     'FontUnits','pixels',...
%     'FontAngle','italic',...
%     'FitBoxToText','off',...
%     'EdgeColor','none');

print('feasi_a','-depsc')
% print('feasi_a','-dtiff')

figure2=figure(2);
plot(0:15, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(0:15, ones(1,16)*mean(rssa), 'k-', 'LineWidth',2)
hold on
plot(ones(1,length(range))*0, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4*2), range, 'b--', 'LineWidth', 2)
hold on
% plot(ones(1,length(range))*(15/4*3), range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,length(range))*15, range, 'b--', 'LineWidth', 2)
% hold on
for p = 1:4
    txt = ['s^B_' num2str(p)];
    text((p-1)*(15/4)+(15/4)/4, ylimd+3, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% % 创建 rectangle
% annotation(figure2,'rectangle',...
%     [0.341666666666667 0.107142857142857 0.173809523809524 0.814285714285715],...
%     'Color',[0.392156862745098 0.831372549019608 0.0745098039215686],...
%     'LineWidth',2,...
%     'LineStyle','--');

% 创建 rectangle
annotation(figure2,'rectangle',...
    [0.688095238095238 0.107285714285714 0.18452380952381 0.814285714285715],...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333],...
    'LineWidth',2);

% % 创建 textbox
% annotation(figure2,'textbox',...
%     [0.686714285714285 0.76825396825397 0.170428571428572 0.147619047619051],...
%     'String',{'Incorrectly','Matched','with $s^B_4$'},...
%     'Interpreter','latex',...
%     'FontWeight','bold',...
%     'FontSize',15,...
%     'FontName','Arial',...
%     'FitBoxToText','off',...
%     'EdgeColor','none');

print('feasi_b','-depsc')
% print('feasi_b','-dtiff')


% Sorting
sorta = sort(rssa);
sortb = sort(rssb);

figure(21)
plot(0:15, sorta, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:15, ones(1,16)*mean(rssa), 'k-', 'LineWidth',2)
hold on
plot(ones(1,length(range))*0, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*15, range, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['t^A_' num2str(p)];
    text((p-1)*(15/4)+(15/4)/4+0.25, ylimd+3, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])

set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('sort_feasi_a','-depsc')
% print('sort_feasi_a','-dtiff')

figure(22)
plot(0:15, sortb, 'ks-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:15, ones(1,16)*mean(rssa), 'k-', 'LineWidth',2)
hold on
plot(ones(1,length(range))*0, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4*2), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*(15/4*3), range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,length(range))*15, range, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['t^B_' num2str(p)];
    text((p-1)*(15/4)+(15/4)/4+0.25, ylimd+3, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([ylimd ylimu])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('sort_feasi_b','-depsc')

