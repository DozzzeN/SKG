close all; clc
textSize = 16;

load ../../corr_mi128_rss_inv.mat
load ../../corr_mi128_rss.mat
allCorr128 = [];
allCorr128 = [allCorr128 inferCorrRSS];
allCorr128 = [allCorr128 imitCorrRSS];
% allCorr128 = [allCorr128 stalkCorrRSS];
allCorr128 = [allCorr128 inv1CorrRSS];
allCorr128 = [allCorr128 inv2CorrRSS];
load ../../corr_si128_rss_inv.mat
load ../../corr_si128_rss.mat
allCorr128 = [allCorr128 inferCorrRSS];
allCorr128 = [allCorr128 imitCorrRSS];
% allCorr128 = [allCorr128 stalkCorrRSS];
allCorr128 = [allCorr128 inv1CorrRSS];
allCorr128 = [allCorr128 inv2CorrRSS];
load ../../corr_mo128_rss_inv.mat
load ../../corr_mo128_rss.mat
allCorr128 = [allCorr128 inferCorrRSS];
allCorr128 = [allCorr128 imitCorrRSS];
% allCorr128 = [allCorr128 stalkCorrRSS];
allCorr128 = [allCorr128 inv1CorrRSS];
allCorr128 = [allCorr128 inv2CorrRSS];
load ../../corr_so128_rss_inv.mat
load ../../corr_so128_rss.mat
allCorr128 = [allCorr128 inferCorrRSS];
allCorr128 = [allCorr128 imitCorrRSS];
% allCorr128 = [allCorr128 stalkCorrRSS];
allCorr128 = [allCorr128 inv1CorrRSS];
allCorr128 = [allCorr128 inv2CorrRSS];

load ../../corr_mi256_rss_inv.mat
load ../../corr_mi256_rss.mat
allCorr256 = [];
allCorr256 = [allCorr256 inferCorrRSS];
allCorr256 = [allCorr256 imitCorrRSS];
% allCorr256 = [allCorr256 stalkCorrRSS];
allCorr256 = [allCorr256 inv1CorrRSS];
allCorr256 = [allCorr256 inv2CorrRSS];
load ../../corr_si256_rss_inv.mat
load ../../corr_si256_rss.mat
allCorr256 = [allCorr256 inferCorrRSS];
allCorr256 = [allCorr256 imitCorrRSS];
% allCorr256 = [allCorr256 stalkCorrRSS];
allCorr256 = [allCorr256 inv1CorrRSS];
allCorr256 = [allCorr256 inv2CorrRSS];
load ../../corr_mo256_rss_inv.mat
load ../../corr_mo256_rss.mat
allCorr256 = [allCorr256 inferCorrRSS];
allCorr256 = [allCorr256 imitCorrRSS];
% allCorr256 = [allCorr256 stalkCorrRSS];
allCorr256 = [allCorr256 inv1CorrRSS];
allCorr256 = [allCorr256 inv2CorrRSS];
load ../../corr_so256_rss_inv.mat
load ../../corr_so256_rss.mat
allCorr256 = [allCorr256 inferCorrRSS];
allCorr256 = [allCorr256 imitCorrRSS];
% allCorr256 = [allCorr256 stalkCorrRSS];
allCorr256 = [allCorr256 inv1CorrRSS];
allCorr256 = [allCorr256 inv2CorrRSS];

load ../../corr_mi512_rss_inv.mat
load ../../corr_mi512_rss.mat
allCorr512 = [];
allCorr512 = [allCorr512 inferCorrRSS];
allCorr512 = [allCorr512 imitCorrRSS];
% allCorr512 = [allCorr512 stalkCorrRSS];
allCorr512 = [allCorr512 inv1CorrRSS];
allCorr512 = [allCorr512 inv2CorrRSS];
load ../../corr_si512_rss_inv.mat
load ../../corr_si512_rss.mat
allCorr512 = [allCorr512 inferCorrRSS];
allCorr512 = [allCorr512 imitCorrRSS];
% allCorr512 = [allCorr512 stalkCorrRSS];
allCorr512 = [allCorr512 inv1CorrRSS];
allCorr512 = [allCorr512 inv2CorrRSS];
load ../../corr_mo512_rss_inv.mat
load ../../corr_mo512_rss.mat
allCorr512 = [allCorr512 inferCorrRSS];
allCorr512 = [allCorr512 imitCorrRSS];
% allCorr512 = [allCorr512 stalkCorrRSS];
allCorr512 = [allCorr512 inv1CorrRSS];
allCorr512 = [allCorr512 inv2CorrRSS];
load ../../corr_so512_rss_inv.mat
load ../../corr_so512_rss.mat
allCorr512 = [allCorr512 inferCorrRSS];
allCorr512 = [allCorr512 imitCorrRSS];
% allCorr512 = [allCorr512 stalkCorrRSS];
allCorr512 = [allCorr512 inv1CorrRSS];
allCorr512 = [allCorr512 inv2CorrRSS];

load ../../corr_mi1024_rss_inv.mat
load ../../corr_mi1024_rss.mat
allCorr1024 = [];
allCorr1024 = [allCorr1024 inferCorrRSS];
allCorr1024 = [allCorr1024 imitCorrRSS];
% allCorr1024 = [allCorr1024 stalkCorrRSS];
allCorr1024 = [allCorr1024 inv1CorrRSS];
allCorr1024 = [allCorr1024 inv2CorrRSS];
load ../../corr_si1024_rss_inv.mat
load ../../corr_si1024_rss.mat
allCorr1024 = [allCorr1024 inferCorrRSS];
allCorr1024 = [allCorr1024 imitCorrRSS];
% allCorr1024 = [allCorr1024 stalkCorrRSS];
allCorr1024 = [allCorr1024 inv1CorrRSS];
allCorr1024 = [allCorr1024 inv2CorrRSS];
load ../../corr_mo1024_rss_inv.mat
load ../../corr_mo1024_rss.mat
allCorr1024 = [allCorr1024 inferCorrRSS];
allCorr1024 = [allCorr1024 imitCorrRSS];
% allCorr1024 = [allCorr1024 stalkCorrRSS];
allCorr1024 = [allCorr1024 inv1CorrRSS];
allCorr1024 = [allCorr1024 inv2CorrRSS];
load ../../corr_so1024_rss_inv.mat
load ../../corr_so1024_rss.mat
allCorr1024 = [allCorr1024 inferCorrRSS];
allCorr1024 = [allCorr1024 imitCorrRSS];
% allCorr1024 = [allCorr1024 stalkCorrRSS];
allCorr1024 = [allCorr1024 inv1CorrRSS];
allCorr1024 = [allCorr1024 inv2CorrRSS];

figure(1)
% for i=1:length(allCorr128)
%     if allCorr128(i) > 0.3
%         allCorr128(i) = allCorr128(i)-0.3;
%     end
% end
% for i=1:length(allCorr256)
%     if allCorr256(i) > 0.3
%         allCorr256(i) = allCorr256(i)-0.3;
%     end
% end
% for i=1:length(allCorr512)
%     if allCorr512(i) > 0.3
%         allCorr512(i) = allCorr512(i)-0.3;
%     end
% end
% for i=1:length(allCorr1024)
%     if allCorr1024(i) > 0.3
%         allCorr1024(i) = allCorr1024(i)-0.3;
%     end
% end
[f1,x1]=ksdensity(allCorr128);
[f2,x2]=ksdensity(allCorr256);
[f3,x3]=ksdensity(allCorr512);
[f4,x4]=ksdensity(allCorr1024);

f1=f1/100;
f2=f2/100;
f3=f3/100;
f4=f4/100;

% f1=f1/length(allCorr128);
% f2=f2/length(allCorr256);
% f3=f3/length(allCorr512);
% f4=f4/length(allCorr1024);

lineWidth=0.7;
plot(x1, f1, 'o-', 'LineWidth',lineWidth, Color="#D95319")
hold on
plot(x2, f2, '*--', 'LineWidth',lineWidth, Color="#EDB120")
hold on
plot(x3, f3, 'd:', 'LineWidth',lineWidth, Color="#7E2F8E")
hold on
plot(x4, f4, '+-.', 'LineWidth',lineWidth, Color="#77AC30")
hold on
% plot(x5, f5, '.-', 'LineWidth',lineWidth, Color="#4DBEEE")
% hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

legend({'N=640', 'N=1280'...
    'N=2560', 'N=5120'},...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northEast')
xlabel('\rho(P^A,P^E)', 'FontSize', 15, 'FontWeight','bold')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')
grid on

xlim([-0.03 0.2]);

print('corrRSS2','-depsc')