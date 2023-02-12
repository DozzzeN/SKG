clear; close all; clc
textSize = 16;
linewid = 1;


% -----------------------------------
% time complexity
epiImp = [0.000995874 0.003988028 0.015946627 0.087574959 0.298328161];
kle = [25 51 102 204 409]*10;
epiImpBi = [0.006317854 0.016201973 0.116560221 0.92144537 3.284394026 18.45500588];
kleBi = [16*4 32*5 64*6 128*7 256*8 512*9];

figure(13)
plot(kle, epiImp, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
plot(kleBi, epiImpBi, 'bo-', 'MarkerFaceColor', 'b', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlim([0 4800]);
ylim([0 25]);
legend('SKG-ED','DP-SKG', 'FontSize',textSize, 'FontWeight','bold', 'Location','northeast')
xlabel('Key Length (bit)', 'FontSize', 15, 'FontWeight','bold')
ylabel('Time (sec.)', 'FontSize',15, 'FontWeight','bold')
print('timeCmplx','-depsc')


epiImpBiDegree = [0.041495323 0.057045221 0.141246557 0.237560272 0.378680229 0.92144537];
kleBiDegree = [4 8 16 32 64 128];
figure(15)
plot(kleBiDegree, epiImpBiDegree, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',2)
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
xlabel('Node Degree', 'FontSize', 15, 'FontWeight','bold')
ylabel('Time (sec.)', 'FontSize',15, 'FontWeight','bold')
print('timeCmplxBiDegree','-depsc')