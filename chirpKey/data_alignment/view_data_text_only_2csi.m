clear;clc
filename = 'test/CSI_214.txt';
% filename= 'modifyseqdata192.168.123.1/CSI_84.txt';
% filename = '192.168.123.2/CSI_37.txt';
% filename= 'simulation/gnuradio/CSI_4.txt';


[csi1,csi2,csi_time, Fs] = read_csi_2(filename);
Fs = round(Fs);
Time = 0:1/Fs:((size(csi1,1)-1)/Fs);
csi_time = Time';

csi_value1 = abs(csi1);
csi_value2 = abs(csi2);

csi_value = (csi_value1+csi_value2)/2;
% csi_value = csi_value2_process;

%%
fig = figure(3)
% subplot(121)
hold on
% surf(1:size(csi_value,1), 1:size(csi_value,2), csi_value', 'EdgeColor','none');
surf(csi_time, 1:size(csi_value,2), csi_value', 'EdgeColor','none');
colormap jet
colorbar
% caxis([0 0.9]);
ylim([1 size(csi_value,2)]);
xlabel('Time(seconds)')
ylabel('Frequency')
grid off
set(gca, 'FontSize', 12,'FontWeight','bold');
set(fig, 'Units', 'pixels', 'Position', [200 200 720 350]);
% axis([1 730 1 52])

%% 
fig = figure(1)
% subplot(121)
hold on
% surf(1:size(csi_value,1), 1:size(csi_value,2), csi_value', 'EdgeColor','none');
surf(csi_time, 1:size(csi_value1,2), csi_value1', 'EdgeColor','none');
colormap jet
colorbar
% caxis([0 0.9]);
ylim([1 size(csi_value1,2)]);
xlabel('Time(seconds)')
ylabel('Frequency')
grid off
set(gca, 'FontSize', 12,'FontWeight','bold');
set(fig, 'Units', 'pixels', 'Position', [200 200 720 350]);

%% 
fig = figure(2)
% subplot(121)
hold on
% surf(1:size(csi_value,1), 1:size(csi_value,2), csi_value', 'EdgeColor','none');
surf(csi_time, 1:size(csi_value2,2), csi_value2', 'EdgeColor','none');
colormap jet
colorbar
% caxis([0 0.9]);
ylim([1 size(csi_value2,2)]);
xlabel('Time(seconds)')
ylabel('Frequency')
grid off
set(gca, 'FontSize', 12,'FontWeight','bold');
set(fig, 'Units', 'pixels', 'Position', [200 200 720 350]);

%% —°‘Ò‘ÿ≤®
% figure(5)
% hold on
% plot(csi_time, abs(csi(:,15)))


m = mean(csi_value,1)';
csi_o = m;
csi_m = m;

% save csi_o csi_o
% save csi_m csi_m
