close all; clc
textSize = 16;

load ../../corr_mi128_inv.mat
load ../../corr_mi128.mat
allCorr128 = [];
allCorr128 = [allCorr128 randomCorr];
allCorr128 = [allCorr128 inferCorr];
allCorr128 = [allCorr128 imitCorr];
allCorr128 = [allCorr128 stalkCorr];
allCorr128 = [allCorr128 inv1Corr];
allCorr128 = [allCorr128 inv2Corr];
load ../../corr_si128_inv.mat
load ../../corr_si128.mat
allCorr128 = [allCorr128 randomCorr];
allCorr128 = [allCorr128 inferCorr];
allCorr128 = [allCorr128 imitCorr];
allCorr128 = [allCorr128 stalkCorr];
allCorr128 = [allCorr128 inv1Corr];
allCorr128 = [allCorr128 inv2Corr];
load ../../corr_mo128_inv.mat
load ../../corr_mo128.mat
allCorr128 = [allCorr128 randomCorr];
allCorr128 = [allCorr128 inferCorr];
allCorr128 = [allCorr128 imitCorr];
allCorr128 = [allCorr128 stalkCorr];
allCorr128 = [allCorr128 inv1Corr];
allCorr128 = [allCorr128 inv2Corr];
load ../../corr_so128_inv.mat
load ../../corr_so128.mat
allCorr128 = [allCorr128 randomCorr];
allCorr128 = [allCorr128 inferCorr];
allCorr128 = [allCorr128 imitCorr];
allCorr128 = [allCorr128 stalkCorr];
allCorr128 = [allCorr128 inv1Corr];
allCorr128 = [allCorr128 inv2Corr];

load ../../corr_mi256_inv.mat
load ../../corr_mi256.mat
allCorr256 = [];
allCorr256 = [allCorr256 randomCorr];
allCorr256 = [allCorr256 inferCorr];
allCorr256 = [allCorr256 imitCorr];
allCorr256 = [allCorr256 stalkCorr];
allCorr256 = [allCorr256 inv1Corr];
allCorr256 = [allCorr256 inv2Corr];
load ../../corr_si256_inv.mat
load ../../corr_si256.mat
allCorr256 = [allCorr256 randomCorr];
allCorr256 = [allCorr256 inferCorr];
allCorr256 = [allCorr256 imitCorr];
allCorr256 = [allCorr256 stalkCorr];
allCorr256 = [allCorr256 inv1Corr];
allCorr256 = [allCorr256 inv2Corr];
load ../../corr_mo256_inv.mat
load ../../corr_mo256.mat
allCorr256 = [allCorr256 randomCorr];
allCorr256 = [allCorr256 inferCorr];
allCorr256 = [allCorr256 imitCorr];
allCorr256 = [allCorr256 stalkCorr];
allCorr256 = [allCorr256 inv1Corr];
allCorr256 = [allCorr256 inv2Corr];
load ../../corr_so256_inv.mat
load ../../corr_so256.mat
allCorr256 = [allCorr256 randomCorr];
allCorr256 = [allCorr256 inferCorr];
allCorr256 = [allCorr256 imitCorr];
allCorr256 = [allCorr256 stalkCorr];
allCorr256 = [allCorr256 inv1Corr];
allCorr256 = [allCorr256 inv2Corr];

load ../../corr_mi512_inv.mat
load ../../corr_mi512.mat
allCorr512 = [];
allCorr512 = [allCorr512 randomCorr];
allCorr512 = [allCorr512 inferCorr];
allCorr512 = [allCorr512 imitCorr];
allCorr512 = [allCorr512 stalkCorr];
allCorr512 = [allCorr512 inv1Corr];
allCorr512 = [allCorr512 inv2Corr];
load ../../corr_si512_inv.mat
load ../../corr_si512.mat
allCorr512 = [allCorr512 randomCorr];
allCorr512 = [allCorr512 inferCorr];
allCorr512 = [allCorr512 imitCorr];
allCorr512 = [allCorr512 stalkCorr];
allCorr512 = [allCorr512 inv1Corr];
allCorr512 = [allCorr512 inv2Corr];
load ../../corr_mo512_inv.mat
load ../../corr_mo512.mat
allCorr512 = [allCorr512 randomCorr];
allCorr512 = [allCorr512 inferCorr];
allCorr512 = [allCorr512 imitCorr];
allCorr512 = [allCorr512 stalkCorr];
allCorr512 = [allCorr512 inv1Corr];
allCorr512 = [allCorr512 inv2Corr];
load ../../corr_so512_inv.mat
load ../../corr_so512.mat
allCorr512 = [allCorr512 randomCorr];
allCorr512 = [allCorr512 inferCorr];
allCorr512 = [allCorr512 imitCorr];
allCorr512 = [allCorr512 stalkCorr];
allCorr512 = [allCorr512 inv1Corr];
allCorr512 = [allCorr512 inv2Corr];

load ../../corr_mi1024_inv.mat
load ../../corr_mi1024.mat
allCorr1024 = [];
allCorr1024 = [allCorr1024 randomCorr];
allCorr1024 = [allCorr1024 inferCorr];
allCorr1024 = [allCorr1024 imitCorr];
allCorr1024 = [allCorr1024 stalkCorr];
allCorr1024 = [allCorr1024 inv1Corr];
allCorr1024 = [allCorr1024 inv2Corr];
load ../../corr_si1024_inv.mat
load ../../corr_si1024.mat
allCorr1024 = [allCorr1024 randomCorr];
allCorr1024 = [allCorr1024 inferCorr];
allCorr1024 = [allCorr1024 imitCorr];
allCorr1024 = [allCorr1024 stalkCorr];
allCorr1024 = [allCorr1024 inv1Corr];
allCorr1024 = [allCorr1024 inv2Corr];
load ../../corr_mo1024_inv.mat
load ../../corr_mo1024.mat
allCorr1024 = [allCorr1024 randomCorr];
allCorr1024 = [allCorr1024 inferCorr];
allCorr1024 = [allCorr1024 imitCorr];
allCorr1024 = [allCorr1024 stalkCorr];
allCorr1024 = [allCorr1024 inv1Corr];
allCorr1024 = [allCorr1024 inv2Corr];
load ../../corr_so1024_inv.mat
load ../../corr_so1024.mat
allCorr1024 = [allCorr1024 randomCorr];
allCorr1024 = [allCorr1024 inferCorr];
allCorr1024 = [allCorr1024 imitCorr];
allCorr1024 = [allCorr1024 stalkCorr];
allCorr1024 = [allCorr1024 inv1Corr];
allCorr1024 = [allCorr1024 inv2Corr];

figure(1)

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
h1=plot(x1, f1, 'square-.', 'LineWidth',lineWidth);
hold on
h2=plot(x2, f2, 'o-', 'LineWidth',lineWidth);
hold on
h3=plot(x3, f3, '*--', 'LineWidth',lineWidth);
hold on
h4=plot(x4, f4, 'd:', 'LineWidth',lineWidth);
hold on
% h5=plot(x5, f5, '+-.', 'LineWidth',lineWidth);
% hold on
% h6=plot(x6, f6, '.-', 'LineWidth',lineWidth);
% hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

% 这样写legend的style才不会错乱
legend([h1 h2 h3 h4], {'M=192', 'M=384'...
    'M=768', 'M=1536'}, ...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northEast')
xlabel('\rho(RO,RO^E)', 'FontSize', 15, 'FontWeight','bold')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')
grid on

ylim([0 0.16]);

print('corrkeys2','-depsc')