clear;clc
filename = 'CSI_sa2.txt';
% filename= 'modifyseqdata192.168.123.1/CSI_84.txt';
% filename = '192.168.123.2/CSI_37.txt';
% filename= 'simulation/gnuradio/CSI_4.txt';


[csi, csi_time, Fs] = read_csi(filename);
Fs = round(Fs);
Time = 0:1/Fs:((size(csi,1)-1)/Fs);
csi_time = Time';

csi_value = abs(csi);
csi_value = csi_value(:,1);
% csi_value = csi_value2_process;
save("csi_sa2.mat", "csi_value", '-mat')

%%
% fig = figure(3)
% % subplot(121)
% hold on
% % surf(1:size(csi_value,1), 1:size(csi_value,2), csi_value', 'EdgeColor','none');
% surf(csi_time, 1:size(csi_value,2), csi_value', 'EdgeColor','none');
% colormap jet
% colorbar
% % caxis([0 0.9]);
% ylim([1 size(csi_value,2)]);
% xlabel('Time(seconds)')
% ylabel('Frequency')
% grid off
% set(gca, 'FontSize', 12,'FontWeight','bold');
% set(fig, 'Units', 'pixels', 'Position', [200 200 720 350]);
% % axis([1 730 1 52])
% 
% 
% %% —°‘Ò‘ÿ≤®
% % figure(5)
% % hold on
% % plot(csi_time, abs(csi(:,15)))
% 
% 
% m = mean(csi_value,1)';
% csi_o = m;
% csi_m = m;
% 
% % save csi_o csi_o
% % save csi_m csi_m

plot(csi_value(:,1))

%%
% load csi_p1.mat
% plot(ans)
% %%
% load csi_p2.mat
% plot(ans)







