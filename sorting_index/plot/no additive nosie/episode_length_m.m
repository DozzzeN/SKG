clear; close all; clc
textSize = 16;
linewid = 1;


% MI 6 7 8 9
% SI 6 7 8 9
% MO 6 7 8 9
% SO 6 7 8 9
% epiImpBMR = [
%     1-0.999902099 0 0 0;
%     1-0.9999472128 0 0 0; 
%     1-0.9997209821 0 0 0;
%     1-0.9994091387 0 0 0
%     ];
epiImpBMR = [
    0 0 0 0;
    1-0.999862 0 0 0; 
    0 0 0 0;
    1-0.9993373326 0 0 0
    ];


epi = [5, 6, 7, 8];

figure(1)
bar(epi, epiImpBMR(1:4,:)', 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold'; 
ax.LineWidth = 1;
ylim([0 0.006])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
grid on
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
    'FontSize',12,...
    'String','0 /',...
    'Position',[4.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.00014 /',...
    'Position',[4.87247591118559 0.000300543259557315 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[5.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.00066',...
    'Position',[5.3 0.000662667399999983 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[5.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[5.9 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[6.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[6.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[6.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[6.9 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[7.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[7.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[7.7 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[7.9 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0 /',...
    'Position',[8.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[8.3 0 0]);

print('epiImpBMRm','-depsc')