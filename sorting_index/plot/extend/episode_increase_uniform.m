clear; close all; clc
textSize = 16;
linewid = 1;

epiIn2 = [0.41674 0.43977 0.45271 0.457158 0.45951 0.45994 0.46128 0.46201 0.46246];
epiIn4 = [0.05187 0.0618 0.06667 0.06884 0.07013 0.07039 0.0709 0.07129 0.07193];
epiIn6 = [0.00567 0.00759 0.00874 0.00906 0.00928 0.00951 0.00948 0.00942 0.00954];
epiIn8 = [0.00062 0.00092 0.00103 0.0011 0.00129 0.00125 0.00119 0.0012 0.00123];
% epiIn10 = [6e-05 0.00011 0.00013 0.00015 0.00017 0.00015 0.00016 0.00013 0.00016];

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
print('epiinc_uni','-depsc')