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
compBMR = [1-0.98828125 1-0.9908854166666666 0 0;
    1-0.9970304528 1-0.935546875 0 0;
    1-0.9454985119 1-0.9981971153846154 0 0;
    1-0.9787946429 1-0.9876302083333334 0 0];

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

ylim([0 0.155])
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
    'String','0.012',...
    'Position',[0.718349392542941 0.0154536897590362 1.4210854715202e-14]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.0091',...
    'Position',[1.03762044407206 0.00911458333333337 1.4210854715202e-14]);

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
    'String','0.003',...
    'Position',[1.61742773355677 0.00296954719999998 1.4210854715202e-14]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.064',...
    'Position',[1.9 0.064453125 0]);

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
    'String','0.055',...
    'Position',[2.7 0.0545014881 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.0018',...
    'Position',[3.04679514034353 0.00600469184430025 1.4210854715202e-14]);

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
    'String','0.021',...
    'Position',[3.7 0.0212053571 0]);

% 创建 text
text('VerticalAlignment','bottom',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',12,...
    'String','0.012',...
    'Position',[4.01009635525765 0.0123697916666666 1.4210854715202e-14]);

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
compBGR = [0.912 1.0455 1.143 2;
    0.996 1.0676 1.143 2;
    0.753 0.9093 1.143 2;
    0.938 1.0822 1.143 2
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

ylim([0 4])
legend('DCT-Quantization', 'DP-SKG', 'SIM-SKG', 'SIM-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BGR (bit per sample)', 'FontSize',15, 'FontWeight','bold')
grid on
for i=1:length(compBGR)
    text(ep(i)-0.3,compBGR(i,1),strcat(num2str(compBGR(i,1), '%.2g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(ep(i)-0.15,0.025+compBGR(i,2),strcat(num2str(compBGR(i,2), '%.2g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(ep(i)+0.05,0.05+compBGR(i,3),strcat(num2str(compBGR(i,3), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(ep(i)+0.275,compBGR(i,4),strcat(num2str(compBGR(i,4), '%.3g')), 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('compvsquantBGR','-depsc')