clear; close all; clc
textSize = 14;

% Actual
% Eavesdropping (n4) IS 256 512 768 1024
% Inference (n3) IS 256 512 768 1024
% Imitation (e1) IS 256 512 768 1024
% Stalking (e2) IS 256 512 768 1024
% inversion 1 IS 256 512 768 1024
% inversion 2 IS 256 512 768 1024

epiImpBMRis = [(256^2-1)/3/256 (512^2-1)/3/512 (768^2-1)/3/768 (1024^2-1)/3/1024; 
    85.34895833 172.00837054 250.228125 340.02473958;
    85.09140625 170.54631696 255.29817708 337.20638021;
    82.83359375 171.29659598 258.82838542 343.09049479;
    85.74192708 169.78097098 261.56901042 339.53580729;
    85.0703125  167.25585938 258.1171875  346.1015625;
    82.1953125  172.0703125  258.7421875  332.98730469
    ];
    
kle = [5, 6, 7, 8];

epiImpBMRis = epiImpBMRis';

figure(31)
bar(kle, epiImpBMRis, 'grouped')
hold on
xticklabels({'256','512','768','1024'})
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 1000])
yticks(0:200:1000)
% legend('RGA Theo.', 'RGA',...
%     'InfA', 'ImA', 'SA', ...
%     'InvA (QP Method)', 'InvA (Iterative Attack)',...
%     'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
legend('Random Guess Attack Theo.', 'Random Guess Attack',...
    'Inference Attack', 'Imitation Attack', 'Stalking Attack', ...
    'Inversion Attack (QP Method)', 'Inversion Attack (Iterative Method)',...
    'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Number of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('Average Difference', 'FontSize',15, 'FontWeight','bold')
grid on

% fsize = 8;
% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),strcat(num2str(epiImpBMRis(i,1), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)-0.1,epiImpBMRis(i,2),strcat(num2str(epiImpBMRis(i,2), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)+0.1,epiImpBMRis(i,3),strcat(num2str(epiImpBMRis(i,3), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
%     text(kle(i)+0.3,epiImpBMRis(i,4),strcat(num2str(epiImpBMRis(i,4), '%.3g')), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fsize)
% end

print('dists2','-depsc')
