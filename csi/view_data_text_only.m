clear;clc
% filename = 'test/CSI_11.txt';
% filename= 'modifyseqdata192.168.123.2/CSI_23.txt';
% filename = '192.168.123.2/CSI_37.txt';
filename= 'simulation/gnuradio/CSI_4.txt';

[csi, csi_time, Fs] = read_csi(filename);
Fs = round(Fs);
Time = 0:1/Fs:((size(csi,1)-1)/Fs);
csi_time = Time';

csi_value = abs(csi);
% csi_value = csi_value2_process;

figure(2)
% subplot(121)
hold on
surf(1:size(csi_value,1), 1:size(csi_value,2), csi_value', 'EdgeColor','none');
colormap jet
colorbar
% caxis([0 4]);
ylim([1 size(csi_value,2)]);
xlabel('Time(seconds)')
ylabel('Frequency')
grid off
set(gca, 'FontSize', 10);


%% —°‘Ò‘ÿ≤®
% figure(5)
% hold on
% plot(csi_time, abs(csi(:,15)))


m = mean(csi_value,1);












