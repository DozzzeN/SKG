close all; clc
textSize = 15;
linewid = 1;

load ../../corr_mi_inv.mat
load ../../corr_mi.mat

figure(1)
nums=min(length(randomCorr), length(inv1Corr));
[f1,x1]=ksdensity(randomCorr(1:nums));
[f2,x2]=ksdensity(inferCorr(1:nums));
[f3,x3]=ksdensity(imitCorr(1:nums));
[f4,x4]=ksdensity(stalkCorr(1:nums));
[f5,x5]=ksdensity(inv1Corr(1:nums));
[f6,x6]=ksdensity(inv2Corr(1:nums));

f1=f1/length(randomCorr);
f2=f2/length(inferCorr);
f3=f3/length(imitCorr);
f4=f4/length(stalkCorr);
f5=f5/length(inv1Corr);
f6=f6/length(inv2Corr);

% f1=f1/100;
% f2=f2/100;
% f3=f3/100;
% f4=f4/100;
% f5=f5/100;
% f6=f6/100;

lineWidth=0.7;
h1=plot(x1, f1, 'square-.', 'LineWidth',lineWidth);
hold on
h2=plot(x2, f2, 'o-', 'LineWidth',lineWidth);
hold on
h3=plot(x3, f3, '*--', 'LineWidth',lineWidth);
hold on
h4=plot(x4, f4, 'd:', 'LineWidth',lineWidth);
hold on
h5=plot(x5, f5, '+-.', 'LineWidth',lineWidth);
hold on
h6=plot(x6, f6, '.-', 'LineWidth',lineWidth);
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

% 这样写legend的style才不会错乱
legend([h1 h2 h3 h4 h5 h6], {'Random Guess Attack',...
        'Inference Attack', 'Imitation Attack', 'Stalking Attack',...
        'QP Method', 'Iterative Method'}, ...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northwest')
xlabel('\rho(RO,RO^E)', 'FontSize', 15, 'FontWeight','bold')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')
grid on

% xlim([-0.1 0.3]);
% ylim([0 12]);

print('corrkeys','-depsc')