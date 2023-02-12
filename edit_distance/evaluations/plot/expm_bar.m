clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% Comparison - DCT quantization vs DP-SKG
%   1e-3       7: 0.0    mobile_indoor
%   3e-3       64(7): 9.42e-6 ;     Stationary indoor 

%   1.5e-2     7: 0.0      mobile_outdoor
%   1e-3       7? 0.0    stationary_outdoor

comp = [1e-3 3e-3 1.5e-3 1e-3; 0 9.42e-6 0 0; 0 3.27e-5 0 0 ];
figure(11)
bar([1 2 3 4], comp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xticklabels({'A','B','C', 'D'})
% xticklabels({'Mobile Indoor', 'Stationary Indoor', 'Mobile Outdoor', 'Stationary Indoor'})

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

% xlim([0 rssLen-1])
ylim([0 3.2e-3])
legend('DCT-Quantization','DP-SKG', 'DP-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('compvsquant','-depsc')

comp = [ 0.9 0.93 0.88 .089 ; 0.857 0.857 0.857 0.857; 5.14 5.14 5.14 5.14];
figure(1111)
bar([1 2 3 4], comp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xticklabels({'A','B','C', 'D'})
% xticklabels({'Mobile Indoor', 'Stationary Indoor', 'Mobile Outdoor', 'Stationary Indoor'})

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

% xlim([0 rssLen-1])
ylim([0 8])
legend('DCT-Quantization','DP-SKG', 'DP-SKGr', 'FontSize', textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BGR', 'FontSize',15, 'FontWeight','bold')
print('compvsquantBGR','-depsc')

% -----------------------------------
% Impact of Entropy-based permutation
% w. perm
% [ 0 9.42e-6  0  0 ]
% w.o. perm
% [ 1.35e-3 1.83e-3 1.12e-5 1.38e-3]

permImp = [0 9.42e-6  0  0 ; 1.35e-3 1.83e-3 1.12e-5 1.38e-3];

figure(12)
bar([1 2 3 4], permImp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xticklabels({'A','B','C', 'D'})
% xticklabels({'Mobile Indoor', 'Stationary Indoor', 'Mobile Outdoor', 'Stationary Indoor'})

row1 = {'Mobile' 'Stationary' 'Mobile' 'Stationary'};
row2 = {'Indoor' 'Indoor' 'Outdoor' 'Outdoor'};
labelArray = [row1; row2]; 
tickLabels = strtrim(sprintf('%s\\newline%s \n', labelArray{:}));
ax.XTickLabel = tickLabels; 

% xlim([0 rssLen-1])
ylim([0 3.2e-3])
legend('w. Entropy Permutation','w.o. Entropy Permutation', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
% xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('permImp','-depsc')

% -----------------------------------
% Impact of LOS (both indoor Stationary)
%         7: 1.91e-4;    8: 4.15e-5;     9: 0.0     10: 0.0  NLOS
%         7: 9.42e-6     8: 0            9: 0.0     10: 0.0  LOS
epiImp = [ 1.91e-4, 4.15e-5, 0, 0; 9.42e-6  0 0 0];
epi = [7, 8, 9, 10];

figure(13)
bar(epi, epiImp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
% ylim([-76 -63])
legend('NLoS', 'LoS', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('losImp','-depsc')

% -----------------------------------
% Impact of episode
%         5: 0.0;    6: 0.0;     7: 0.0     8: 0.0  mobile_indoor
%         5:         6:1.23e-5 1.86e-3  7: 0.0  2.77e-4    8: 3e-5 9: 5e-6  stationary_indoor 
%    64(5):1.24e-3   64(6): 1.37e-4   64(7): 9.42e-6 ;   64(8): 0.0   Stationary indoor 

%         5:0.0      6: 0.0      7: 0.0     8: 0.0  mobile_outdoor
%         5:2.79e-4  6: 7.6e-6   7: 0.0     8: 0.0  stationary_outdoor
epiImp = [0, 0, 0, 0; 1.24e-3, 1.37e-4  9.42e-6  0;  0 0 0 0;  2.79e-4 7.6e-6 0 0];
epi = [5, 6, 7, 8];

figure(1)
bar(epi, epiImp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
% ylim([-76 -63])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImp','-depsc')

figure(111)
bar(epi, epiImp(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
ylim([0 1.5e-3])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpIndoor','-depsc')

figure(112)
bar(epi, epiImp(3:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
ylim([0 1.5e-3])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpOutdoor','-depsc')


figure(113)
bar(epi, 6./[5, 6, 7, 8], 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
ylim([0 1.3])
% legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('epiImpKGR','-depsc')


% -----------------------------------
% Impact of episode
% 16: (7): 0;    32 (7): 0;     64 (7): 0;         128 (7): 0    mobile indoor
% 16(7):3.42e-5  32(7):1.37e-4  64(7): 2.77e-4 ;   128: 7.3e-4   Stationary indoor 
% 16(7):0        32(7):4.3e-6   64(7): 9.42e-6 ;   128: 3.77e-5   Stationary indoor 

% 16: 0          32: 0          64:   0            128:  0       mobile outdoor
% 16:0           32: 0          64:   0            128:  3.42e-5       Stationary outdoor

kleImp = [0 0 0 0; 0 4.3e-6 9.42e-6 3.77e-5; 0 0 0 0; 0 0 0 3.4e-6 ];
kle = [16, 32, 64, 128];
figure(2)
bar(epi, kleImp, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'16','32','64','128'})
% xlim([0 rssLen])
ylim([0 4e-5])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BGR', 'FontSize',15, 'FontWeight','bold')
print('kleImp','-depsc')


figure(211)
bar(epi, kleImp(1:2,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'16','32','64','128'})
% xlim([0 rssLen])
ylim([0 5e-5])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('kleImpIndoor','-depsc')

figure(212)
bar(epi, kleImp(3:4,:), 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'16','32','64','128'})
% xlim([0 rssLen])
ylim([0 5e-5])
legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
print('kleImpOutdoor','-depsc')


figure(213)
bar(epi, [4, 5, 6, 7]/7, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% xlim([0 rssLen-1])
ylim([0 1.3])
% legend('Mobile','Stationary', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BGR', 'FontSize',15, 'FontWeight','bold')
print('kleImpKGR','-depsc')

