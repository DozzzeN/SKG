clear; close all; clc
textSize = 16;
linewid = 1;


% Eavesdropping (n4) IS 3 4 5 6
% Inference (n3) IS 3 4 5 6
% Imitation (e1) IS 3 4 5 6
% Stalking (e2) IS 3 4 5 6

epiImpBMRis = [
    1-0.4979003906 1-0.497578125 1-0.5006347656 1-0.5041666667;
    1-0.4966471354 1-0.4984960937 1-0.4979003906 1-0.5029947917;
    1-0.5013346354 1-0.5023632812 1-0.4995605469 1-0.5022786458;
    1-0.4986653646 1-0.4992382813 1-0.5019775391 1-0.4976236979];
kle = [3, 4, 5, 6];

figure(31)
bar(kle, epiImpBMRis, 'grouped')
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
ylim([0 0.9])
yticks(0:0.1:0.9)
legend('Random Guess Attack','Inference Attack', 'Imitation Attack', 'Stalking Attack', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('BMR', 'FontSize',15, 'FontWeight','bold')
grid on

% for i=1:length(epiImpBMRis)
%     text(kle(i)-0.3,epiImpBMRis(i,1),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)-0.1,epiImpBMRis(i,2),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.1,epiImpBMRis(i,3),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i)+0.3,epiImpBMRis(i,4),num2str(epiImpBMRis(i), '%.4g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end

print('attacksBMR','-depsc')
