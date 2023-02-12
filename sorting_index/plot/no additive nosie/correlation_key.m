close all; clc
textSize = 16;
linewid = 1;

figure(1)
% f1出现bug
[f1,x1]=ksdensity(legiCorr);
[f2,x2]=ksdensity(randomCorr);
[f3,x3]=ksdensity(inferCorr);
[f4,x4]=ksdensity(imitCorr);
[f5,x5]=ksdensity(stalkCorr);
f1=f1/length(legiCorr);
f2=f2/length(randomCorr);
f3=f3/length(inferCorr);
f4=f4/length(imitCorr);
f5=f5/length(stalkCorr);

legiCorrb(1)=0;  % 保证能连成一条曲线
plot(ones(length(legiCorrb)), legiCorrb, 'o-', 'LineWidth',1)
hold on
plot(x2, f2, '*--', 'LineWidth',1)
hold on
plot(x3, f3, 'd:', 'LineWidth',1)
hold on
plot(x4, f4, '+-.', 'LineWidth',1)
hold on
plot(x5, f5, 'x-', 'LineWidth',1)
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

legend('Original', 'Random Guess Attack',...
        'Inference Attack', 'Imitation Attack', 'Stalking Attack', ...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northwest')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')
xlabel('...', 'FontSize', 15, 'FontWeight','bold')
grid on

print('corrkeys','-depsc')