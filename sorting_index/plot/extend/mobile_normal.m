clear; close all; clc
textSize = 16;
linewid = 1;

epiIn2 = [0.90992, 0.98298, 0.99673, 0.99943, 0.99994, 0.99998, 1.0, 1.0, 1.0];
epiIn4 = [0.58572, 0.90397, 0.9809, 0.99619, 0.9992, 0.99991, 0.99996, 1.0, 1.0];
epiIn6 = [0.27969, 0.78556, 0.95493, 0.99123, 0.99814, 0.99954, 0.9999, 0.99997, 1.0];
epiIn8 = [0.10445, 0.64136, 0.91778, 0.98261, 0.99654, 0.99946, 0.99988, 0.99997, 1.0];

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
print('epiinc_norm_mobile','-depsc')