clear; close all; clc
textSize = 16;

ep = [1 2 3 4];
% MI DCT DP ISm
% SI DCT DP ISm
% MO DCT DP ISm
% SO DCT DP ISm
% compBMR = [1-0.9846540178571429 1-0.9908854166666666 1-0.9997558594 1-0.9995605469;
%     1-0.9935825892857143 1-0.935546875 1-0.9998604911 1-0.9979870855;
%     1-0.9239211309523809 1-0.9981971153846154 1-0.9994303385 0;
%     1-0.9725632440476191 1-0.9876302083333334 1-0.9996744792 0];
% compBMR = [1-0.9846540178571429 1-0.9908854166666666 0;
%     1-0.9935825892857143 1-0.935546875 0;
%     1-0.9239211309523809 1-0.9981971153846154 0;
%     1-0.9725632440476191 1-0.9876302083333334 0];
% SKYGlow.c
compBMR = [1-0.9794270833 1-0.9904296875 0 0;
    1-0.9924958882 1-0.8419202303 0 0;
    1-0.921484375 1-0.9509765625 0 0;
    1-0.9625488281 1-0.9076660156 0 0];

figure(11)
bar(ep, compBMR, 'grouped')
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
% for i=1:length(compBMR)
%     text(ep(i)-0.3,compBMR(i,1),num2str(compBMR(i,1), '%.2g'), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(ep(i)-0.1,compBMR(i,2),num2str(compBMR(i,2), '%.2g'), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(ep(i)+0.1,compBMR(i,3),strcat(num2str(compBMR(i,3), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(ep(i)+0.3,compBMR(i,4),strcat(num2str(compBMR(i,4), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end
% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.021',...
    'Position',[0.718349392542941 0.0247897841698796 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.0096',...
    'Position',[1.04679514034353 0.012582360692771 0]);

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
    'String','0.0075',...
    'Position',[1.6 0.0075041118 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.16',...
    'Position',[1.9 0.1580797697 0]);

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
    'String','0.079',...
    'Position',[2.7 0.078515625 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.049',...
    'Position',[3 0.0490234375 0]);

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
    'String','0.037',...
    'Position',[3.63 0.0374511719 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.092',...
    'Position',[3.9 0.0923339844 0]);

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

print('compvsquantBMR','-depsc')

% compBGR = [0.9 0.93 0.88 0.89; 
%     0.857 0.857 0.857 0.857; 
%     1.4 1.4 1.4 1.4; 
%     4.2 4.2 4.2 4.2];
% compBGR = [0.90 1.0455 1.143;
%     0.94 1.0676 1.143;
%     0.87 0.9093 1.143;
%     0.91 1.0822 1.143
%     ];
% SKYGlow.c
compBGR = [0.7749446247360017 1.5826893353941268 2 2.9;
    0.9606965174129353 1.5511641791044777 2 2.9;
    0.6994810971089696 1.5183867141162515 2 2.9;
    0.8787500557214818 1.6576600677724274 2 2.9
    ];

figure(1111)
bar(ep, compBGR, 'grouped')
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

ylim([0 6])
legend('DCT-Quantization', 'DP-SKG', 'SIM-SKG', 'SIM-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BGR (bit per sample)', 'FontSize',15, 'FontWeight','bold')
grid on
for i=1:length(compBGR)
    text(ep(i)-0.32,compBGR(i,1),strcat(num2str(compBGR(i,1), '%.2g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(ep(i)-0.15,0.025+compBGR(i,2),strcat(num2str(compBGR(i,2), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(ep(i)+0.1,0.05+compBGR(i,3),strcat(num2str(compBGR(i,3), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(ep(i)+0.3,0.05+compBGR(i,4),strcat(num2str(compBGR(i,4), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('compvsquantBGR','-depsc')