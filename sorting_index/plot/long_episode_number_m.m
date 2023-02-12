clear; close all; clc
textSize = 16;
linewid = 1;


% MI 256 384 512 640
% SI 256 384 512 640
% MO 256 384 512 640
% SO 256 384 512 640
kleImpBMR = [
   0 0 0 0 0 0 0; 
   0 0 0 0 0 0 0;
   0 0 0 0 0 1-0.9984809028 1-0.9967730978; 
   0 0 0 0 0 0 0
   ];
epi = [4, 5, 6, 7, 8, 9, 10];

figure(1)
bar(epi, kleImpBMR(1:4,:)', 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xticklabels({'32','64','128','256','384','512','640'})
ylim([0 0.006])
legend('Mobile Indoor','Stationary Indoor', 'Mobile Outdoor', 'Stationary Outdoor', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('# of Episodes', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR (%)', 'FontSize',15, 'FontWeight','bold')

for i=1:length(kleImpBMR)
    text(epi(i)-0.3,kleImpBMR(1,i),strcat(num2str(kleImpBMR(1,i), '%.2g'),"/"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)-0.1,kleImpBMR(2,i),strcat(num2str(kleImpBMR(2,i), '%.2g'),"/"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.1,kleImpBMR(3,i),strcat(num2str(kleImpBMR(3,i), '%.2g'),"/"), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
    text(epi(i)+0.3,kleImpBMR(4,i),num2str(kleImpBMR(4,i), '%.2g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

print('kleImpBMRLm','-depsc')