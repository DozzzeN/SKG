close all; clc
textSize = 16;

load ../../corr_mo_rss_inv.mat
load ../../corr_mo_rss.mat
figure(1)
nums=min(length(inferCorrRSS), length(inv1CorrRSS));
[f1,x1]=ksdensity(inferCorrRSS(1:nums));
[f2,x2]=ksdensity(imitCorrRSS(1:nums));
[f3,x3]=ksdensity(stalkCorrRSS(1:nums));
[f4,x4]=ksdensity(inv1CorrRSS(1:nums));
[f5,x5]=ksdensity(inv2CorrRSS(1:nums));

f1=f1/length(inferCorrRSS);
f2=f2/length(imitCorrRSS);
f3=f3/length(stalkCorrRSS);
f4=f4/length(inv1CorrRSS);
f5=f5/length(inv2CorrRSS);

x3 = x3/2.5;
f3 = f3*2.5;

lineWidth=0.7;
plot(x1, f1, 'o-', 'LineWidth',lineWidth, Color="#D95319")
hold on
plot(x2, f2, '*--', 'LineWidth',lineWidth, Color="#EDB120")
hold on
plot(x3, f3, 'd:', 'LineWidth',lineWidth, Color="#7E2F8E")
hold on
plot(x4, f4, '+-.', 'LineWidth',lineWidth, Color="#77AC30")
hold on
plot(x5, f5, '.-', 'LineWidth',lineWidth, Color="#4DBEEE")
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

% xlim([-0.62 0.42]);
% xlim([-0.07 0.3]);
ylim([0 0.26]);
legend('Inference Attack', 'Imitation Attack', 'Stalking Attack', ...
        'QP Method', 'Iterative Method',...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northEast')
xlabel('\rho(P^A,P^E)', 'FontSize', 15, 'FontWeight','bold')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')
grid on

print('corrRSS','-depsc')