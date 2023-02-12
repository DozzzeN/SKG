clear; close all; clc
textSize = 16;
linewid = 1;

epiIn2 = [0.38882 0.3941 0.39887 0.40014 0.40382 0.40535 0.40387 0.40614 0.40731];
epiIn4 = [0.04739 0.04918 0.04981 0.04913 0.04955 0.04743 0.04553 0.04616 0.04608];
epiIn6 = [0.005 0.00573 0.0055 0.00516 0.00529 0.00462 0.00433 0.0047 0.0049];
epiIn8 = [0.00046 0.00069 0.00053 0.00047 0.00052 0.00049 0.00043 0.00048 0.00048];

kle = [1, 2, 3, 4, 5, 6, 7, 8, 9];

figure(1)

semilogy(kle, epiIn2, 'o-', 'LineWidth',2)
hold on
semilogy(kle, epiIn4, '+--', 'LineWidth',2)
hold on
semilogy(kle, epiIn6, '*:', 'LineWidth',2)
hold on
semilogy(kle, epiIn8, 'x-.', 'LineWidth',2)
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
ylim([3e-4 10]);
legd=legend('n=2', 'n=4', 'n=6', 'n=8', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast');
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
set(legd,'NumColumns',2,'FontWeight','bold','FontSize',16);
print('epiinc_norm','-depsc')