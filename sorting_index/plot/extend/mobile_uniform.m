clear; close all; clc
textSize = 16;
linewid = 1;

epiIn2 = [0.91222, 0.98424, 0.99721, 0.99931, 0.99992, 0.99999, 1.0, 1.0, 1.0];
epiIn4 = [0.57763, 0.90807, 0.98128, 0.99661, 0.99937, 0.99992, 0.99999, 1.0, 1.0];
epiIn6 = [0.25782, 0.78861, 0.95589, 0.99134, 0.99833, 0.99958, 0.99996, 1.0, 0.99999];
epiIn8 = [0.08667, 0.63882, 0.9198, 0.98402, 0.99713, 0.9996, 0.99981, 0.99998, 1.0];

kle = [1, 2, 3, 4, 5, 6, 7, 8, 9];

figure(1)

plot(kle, epiIn2, 'o-', 'LineWidth',2)
hold on
plot(kle, epiIn4, '+--', 'LineWidth',2)
hold on
plot(kle, epiIn6, '*:', 'LineWidth',2)
hold on
plot(kle, epiIn8, 'x-.', 'LineWidth',2)
hold on
% semilogy(kle, epiIn10, 'd-', 'LineWidth',2)
% hold on

ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;
xlim([0.5 9.5]);
ylim([0 1.1]);
legd=legend('n=2', 'n=4', 'n=6', 'n=8', 'FontSize',textSize, 'FontWeight','bold', 'Location','southeast');
xlabel('Episode Length', 'FontSize', 15, 'FontWeight','bold')
ylabel('Prob.', 'FontSize',15, 'FontWeight','bold')
grid on
ax.XGrid = 'on';
ax.YMinorGrid = 'off';
ax.YMinorTick = "off";

% for i=1:length(epiIn2)
%     text(kle(i),epiIn2(i),num2str(epiIn2(i), '%.3g'), 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')

%     text(kle(i),epiIn2(i),num2str(epiImpbm(i), '%.2g'), 'FontWeight','bold', 'Color','#0072BD', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     text(kle(i),epiIn4(i),num2str(epilmpdct(i), '%.2g'), 'FontWeight','bold', 'Color', '#D95319', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%      text(kle(i),epiIn2(i),num2str(epiImpis(i), '%.2g'), 'FontWeight','bold', 'Color', '#EDB120', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
% end
set(legd,'NumColumns',1,'FontWeight','bold','FontSize',16);
print('epiinc_uni_mobile','-depsc')