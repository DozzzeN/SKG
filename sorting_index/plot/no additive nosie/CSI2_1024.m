clear; close all; clc
textSize = 16;
linewid = 1;

% MI DCT DP SIM SIMr
% SI DCT DP SIM SIMr
% MO DCT DP SIM SIMr
% SO DCT DP SIM SIMr

epiImpBMR = [
    1-0.9353515625 1-0.9756510417 0 0;
    1-0.989453125 1-0.8271949405 0 0;
    1-0.9220703125 1-0.9798900463 0 0;
    1-0.9921875 1-0.9454495614 0 0
    ];
epi = [1 2 3 4];

figure(111)
bar(epi, epiImpBMR, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

ylim([0 0.3])
legend('DCT-Quantization', 'DP-SKG', 'SIM-SKG', 'SIM-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
grid on
% for i=1:length(epiImpBMR)
%     text(epi(i)-0.3,epiImpBMR(i,1),num2str(epiImpBMR(i,1), '%.2g'), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)-0.1,epiImpBMR(i,2),num2str(epiImpBMR(i,2), '%.2g'), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)+0.1,epiImpBMR(i,3),strcat(num2str(epiImpBMR(i,3), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(epi(i)+0.3,epiImpBMR(i,4),strcat(num2str(epiImpBMR(i,4), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.065',...
    'Position',[0.721407624633431 0.0640460278614458 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.024',...
    'Position',[1.01009635525765 0.0243489582999999 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[1.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[1.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.011',...
    'Position',[1.64495182237118 0.010546875 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.17',...
    'Position',[1.9 0.1728050595 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[2.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[2.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.078',...
    'Position',[2.70917469627147 0.0779296875 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.02',...
    'Position',[2.96422287390029 0.0201099537 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[3.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[3.3 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.0078',...
    'Position',[3.58990364474235 0.0078125 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.055',...
    'Position',[3.9 0.0545504386 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[4.1 0 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0',...
    'Position',[4.3 0 0]);

print('epiCSIBMR','-depsc')

% MI DCT DP SIM SIMr
% SI DCT DP SIM SIMr
% MO DCT DP SIM SIMr
% SO DCT DP SIM SIMr

epiImp = [
    0.49849068387634016 1.8720799500312304 2 2.9;
    0.7779196704428424 1.4744694960212201 2 2.9;
    0.8396685082872928 1.6744128553770086 2 2.9;
    0.5143782908059943 1.6763061968408262 2 2.9
    ];

figure(113)
bar(epi, epiImp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels;

ylim([0 6])
legend('DCT-Quantization', 'DP-SKG', 'SIM-SKG', 'SIM-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BGR (bit per sample)', 'FontSize',15, 'FontWeight','bold')
grid on
for i=1:length(epiImp)
    text(epi(i)-0.32,epiImp(i,1),strcat(num2str(epiImp(i,1), '%.2g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)-0.15,0.025+epiImp(i,2),strcat(num2str(epiImp(i,2), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.1,0.05+epiImp(i,3),strcat(num2str(epiImp(i,3), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.3,0.05+epiImp(i,4),strcat(num2str(epiImp(i,4), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('epiCSIBGR','-depsc')