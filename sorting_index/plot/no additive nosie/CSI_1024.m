clear; close all; clc
textSize = 16;
linewid = 1;

% ISm-CSI 3 4 5 6
% ISm-RSS 3 4 5 6

epiImpBMR = [
    1-0.9934570312 1-0.9998779297 0 0;
    1-0.9569839015 1-0.9681640625 0 0;
    ];
epi = [3 4 5 6];

figure(111)
bar(epi, epiImpBMR(1:2,:)', 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 0.1]);
legend('SIM-SKG with CSI', 'SIM-SKG with RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR (%)', 'FontSize',15, 'FontWeight','bold')
grid on
% for i=1:length(epiImpBMR)
%     text(epi(i)-0.15,epiImpBMR(1,i),strcat(num2str(epiImpBMR(1,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)+0.15,epiImpBMR(2,i),strcat(num2str(epiImpBMR(2,i), '%.2g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0.0065 /',...
    'Position',[2.77679394338381 0.00674417604346082 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0.043',...
    'Position',[3.15 0.0430160985 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','1e-4 /',...
    'Position',[3.85 0.000122070299999999 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0.032',...
    'Position',[4.15 0.0318359375 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[4.85 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0',...
    'Position',[5.15 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[5.85 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0',...
    'Position',[6.15 0 0]);

print('epiCSIBMR','-depsc')

% epiImp = [
%     3*7/5 3*7/6 3*7/7 3*7/8;
%     7/5 7/6 7/7 7/8
%     ];
epiImp = [
    10/3*0.9569839015 10/4*0.9998779297 10/5 10/6;
    10/3*0.9569839015 10/4*0.9681640625 10/5 10/6
    ];
epiImp = epiImp.';
figure(113)
bar(epi, epiImp(1:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 5])
legend('SIM-SKG with CSI', 'SIM-SKG with RSS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BGR (bit per sample)', 'FontSize',15, 'FontWeight','bold')
grid on

for i=1:length(epiImp)
    text(epi(i)-0.15,epiImp(i,1),strcat(num2str(epiImp(i,1), '%.2g'), " /"), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.15,epiImp(i,2),num2str(epiImp(i,2), '%.2g'), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('epiCSIBGR','-depsc')