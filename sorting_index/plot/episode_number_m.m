clear; close all; clc
textSize = 16;
linewid = 1;


% MI 8 16 32 64
% SI 8 16 32 64
% MO 8 16 32 64
% SO 8 16 32 64
% kleImpBMR = [
%     0 0 0 1-0.9995605469; 
%     0 0 0 1-0.9979870855;
%     0 0 0 0; 
%     0 0 0 0
%    ];
kleImpBMR = [
    0 0 0 0; 
    0 0 0 0;
    0 0 0 0; 
    0 0 0 0
   ];
epi = [4, 5, 6, 7];

figure(1)
bar(epi, kleImpBMR(1:4,:)', 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'32', '64','128','256'})
ylim([0 0.0035])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR (%)', 'FontSize',15, 'FontWeight','bold')

for i=1:length(kleImpBMR)
    text(epi(i)-0.3,kleImpBMR(1,i),strcat(num2str(kleImpBMR(1,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)-0.1,kleImpBMR(2,i),strcat(num2str(kleImpBMR(2,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.1,kleImpBMR(3,i),strcat(num2str(kleImpBMR(3,i), '%.2g')," /"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.3,kleImpBMR(4,i),num2str(kleImpBMR(4,i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('kleImpBMRm','-depsc')