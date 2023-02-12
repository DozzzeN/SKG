clear; close all; clc
textSize = 14;
linewid = 1;

points = [];
lowers = [];
for i=1:65
    lowers(i) = 2^(i-1);
    points(i) = factorial(i-1);
end
figure(1)
plot([1:65], points, 'k--', 'LineWidth',2)
hold on
plot([1:65], lowers, 'r-', 'LineWidth',2)
hold on

ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.YScale='log';
ax.YTick=[0, 1e+20, 1e+40, 1e+60, 1e+80, 1e+100, 1e+120];

ylim([0, 1e+120]);
legend('M_{guess}^{BM}','2^M', 'FontSize',textSize, 'FontWeight','bold', 'Location','northwest')
xlabel('M', 'FontSize', 15, 'FontWeight','bold')
ylabel('The number of guesses', 'FontSize',15, 'FontWeight','bold')
grid on

print('nAllPlot_dp','-depsc')