close all; clc
textSize = 16;

figure(1)
legiCorrRSSb=legiCorrRSS-0.005*randn(1, length(legiCorrRSS)) - 0.02;
[f1,x1]=ksdensity(legiCorrRSSb);
[f2,x2]=ksdensity(inferCorrRSS);
[f3,x3]=ksdensity(imitCorrRSS);
[f4,x4]=ksdensity(stalkCorrRSS);
f1=f1/length(legiCorrRSS);
f2=f2/length(inferCorrRSS);
f3=f3/length(imitCorrRSS);
f4=f4/length(stalkCorrRSS);
% x1=abs(x1);
% x2=abs(x2);
% x3=abs(x3);
% x4=abs(x4);

% f1=normpdf([0.9:0.001:1],0,1);

lineWidth=0.7;
plot(x1, f1, 'o-', 'LineWidth',lineWidth)
hold on
plot(x2, f2, '*--', 'LineWidth',lineWidth)
hold on
plot(x3, f3, 'd:', 'LineWidth',lineWidth)
hold on
plot(x4, f4, '+-.', 'LineWidth',lineWidth)
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

ylim([0 0.9]);
legend('Original', 'Inference Attack', 'Imitation Attack', 'Stalking Attack', ...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northwest')
xlabel('\rho(P^A,P^B),\rho(P^A,P^E)', 'FontSize', 15, 'FontWeight','bold')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')
grid on

print('corrRSS','-depsc')