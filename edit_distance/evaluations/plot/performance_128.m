clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Comparison - DCT quantization DP-SKG vs SKG-ED
%   mobile_indoor
%   stationary_indoor 

%   mobile_outdoor
%   stationary_outdoor

% comp = [1-0.995912666 1-0.999522293 1-0.960069444 1-0.988794643;
%     0 1-0.999666889 0 0;
%     0 9.42e-6 0 0;
%     0 3.27e-5 0 0];
% figure(11)
% bar([1 2 3 4], comp, 'grouped')  % label名；数据；分组style
% hold on
% ax = gca;
% ax.YAxis.FontSize = 13;
% ax.XAxis.FontSize = 13;
% ax.YAxis.FontWeight = 'bold';
% ax.XAxis.FontWeight = 'bold';
% 
% row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
% row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
% labelArray = [row1; row2]; 
% tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
% ax.XTickLabel = tickLabels; 
% 
% ylim([0 0.055])
% legend('DCT-Quantization','SKG-ED', 'DP-SKG', 'DP-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
% ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
% print('compvsquant','-depsc')

comp = [1-0.995912666 1-0.999522293 1-0.960069444 1-0.988794643;
    0 1-0.999666889 0 0];
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

ylim([0 0.055])
legend('DCT-Quantization','SKG-ED', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('compvsquantBMR','-depsc')

%   mobile_indoor
%   stationary_indoor 

%   mobile_outdoor
%   stationary_outdoor
comp = [1-0.708609272 1-0.943949045 1-0.055555556 1-0.32; 0 1-0.993630573 0 0];
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

ylim([0 1.25])
legend('DCT-Quantization','SKG-ED', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
ylabel('KMR', 'FontSize',15, 'FontWeight','bold')
print('compvsquantKMR','-depsc')