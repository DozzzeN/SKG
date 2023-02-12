clear; close all; clc
textSize = 16;
linewid = 1;

load data_static_indoor_1.mat

bin = 30;

% Original RSS Sequence
rssa = A(1:15000,1);
rssb = A(:,2);

figure(1)
% histogram(rssa - mean(rssa));
histogram(rssa - mean(rssa), bin);
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
% ylim([0 3300]);
xlim([-10 10]);
xlabel('RSS (dB)', 'FontSize', 15, 'FontWeight','bold')
ylabel('Frequency', 'FontSize',15, 'FontWeight','bold')

print('oriHist','-depsc')

step = 3;

% RIC-SKGa
addnoise = unifrnd(-std(rssa)*step,std(rssa)*step,length(rssa),1);
% addnoise = normpdf([1:length(rssa)],0,std(rssa)*step);
addrssa = (rssa - mean(rssa)) + addnoise;

figure(2)
% histogram(addrssa);
histogram(addrssa, bin);
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ylim([0 1400]);
xlim([-10 10]);
xlabel('RSS (dB)', 'FontSize', 15, 'FontWeight','bold')
ylabel('Frequency', 'FontSize',15, 'FontWeight','bold')

print('addHist','-depsc')

% RIC-SKGm
step = 1;
% mulnoise = unifrnd(-std(rssa)*step,std(rssa)*step,length(rssa),length(rssa));
mulnoise = unifrnd(0,0.05,length(rssa),length(rssa));
mulrssa = (rssa - mean(rssa))' * mulnoise;

figure(3)
histogram(mulrssa, bin);
% histogram(mulrssa);
hold on
ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';

ylim([0 2000]);
xlim([-10 10]);
xlabel('RSS (dB)', 'FontSize', 15, 'FontWeight','bold')
ylabel('Frequency', 'FontSize',15, 'FontWeight','bold')

print('mulHist','-depsc')