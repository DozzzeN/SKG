clear; close all; clc
textSize = 16;
linewid = 1;

epiIn2 = [0.39877, 0.39975, 0.40493, 0.40771, 0.41282, 0.41156, 0.41144, 0.40942, 0.41201];
epiIn4 = [0.04801, 0.05136, 0.05028, 0.04991, 0.04979, 0.0476, 0.04781, 0.0499, 0.04884];
epiIn6 = [0.00507, 0.00619, 0.00545, 0.0059, 0.00486, 0.00548, 0.00463, 0.0048, 0.00468];
epiIn8 = [0.00055, 0.00068, 0.00069, 0.0006, 0.00053, 0.00043, 0.00051, 0.00051, 0.00046];

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
print('epiinc_norm_static','-depsc')