clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Comparison - DCT quantization vs SKG-ED
%   mobile_indoor
%   stationary_indoor 

%   mobile_outdoor
%   stationary_outdoor

comp = [1-0.989119224 0 1-0.976996528 1-0.982767857; 0 0 0 0];
figure(11)
bar([1 2 3 4], comp, 'grouped')  % label名；数据；分组style
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

ylim([0 0.03])
legend('DCT-Quantization','SKG-ED', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('compvsquant','-depsc')

%   mobile_indoor
%   stationary_indoor 

%   mobile_outdoor
%   stationary_outdoor
comp = [1-0.904290429 1-0.99044586 1-0.555555556 1-0.771428571; 0 0 0 0];
figure(1111)
bar([1 2 3 4], comp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

ylim([0 0.6])
legend('DCT-Quantization','SKG-ED', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('compvsquantBGR','-depsc')