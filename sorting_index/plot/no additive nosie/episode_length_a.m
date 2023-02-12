clear; close all; clc
textSize = 16;
linewid = 1;


% MI 6 7 8 9
% SI 6 7 8 9
% MO 6 7 8 9
% SO 6 7 8 9
epiImpBMR = [
    1-0.9998486683 0 0 0;
    1-0.9998718026 0 0 0;
    1-0.9993303571 0 0 0;
    0 0 0 0
    ];

epi = [5, 6, 7, 8];

figure(1)
bar(epi, epiImpBMR(1:4,:)', 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold'; 
ylim([0 0.0015])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR (%)', 'FontSize',15, 'FontWeight','bold')

% for i=1:length(epiImpBMR)
%     text(epi(i)-0.3,epiImpBMR(1,i),strcat(num2str(epiImpBMR(1,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)-0.1,epiImpBMR(2,i),strcat(num2str(epiImpBMR(2,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)+0.1,epiImpBMR(3,i),strcat(num2str(epiImpBMR(3,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)+0.3,epiImpBMR(4,i),num2str(epiImpBMR(4,i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0.00015 /',...
    'Position',[4.74587348135735 0.000209831700000035 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0.00013',...
    'Position',[4.79907834101383 0.00014619740000003 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0.00067',...
    'Position',[5.1 0.000669642900000045 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0',...
    'Position',[5.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[5.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[5.9 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[6.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0',...
    'Position',[6.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[6.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[6.9 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[7.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0',...
    'Position',[7.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[7.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[7.9 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0 /',...
    'Position',[8.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'String','0',...
    'Position',[8.3 0 0]);

print('epiImpBMRa','-depsc')