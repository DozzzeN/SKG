clear all; close all; clc

load ../data_mobile_indoor_2.mat

% Original RSS Sequence
ind = 2145:2145+19;
rssa = A(ind,1);
rssb = A(ind,2);

insertind = 236:240;
rssinsert = A(insertind,1);

xliml = -1;
xlimr = 20;

figure(1)
plot(0:19, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:19, ones(1,20)*mean(rssa), 'k-', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-80 -55])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('ori_a','-dtiff')

figure(2)
plot(0:19, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(0:19, ones(1,20)*mean(rssa), 'k-', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-80 -55])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('ori_b','-dtiff')

% Add Noise
% mulnoise = unifrnd(0,3,length(rssa),length(rssa));
% mulrssa = (rssa - mean(rssa))' * mulnoise;
% mulrssb = (rssb - mean(rssb))' * mulnoise;
mulrssa = [-12.2234688815244	-21.9214119372032	-2.48446729698878	6.66664099605350	4.50485786331815	23.2810038325226	16.2309244095491	-28.1748322624632	10.5685866489858	4.90226693633466	2.62571451285750	-18.7869887043907	13.7782470211502	9.21747164865053	12.9998693570170	10.7206035902625	-10.5991068430802	30.4621105802946	2.86660583011881	11.3383729444638];
mulrssb = [-15.3487602093957	-21.1357663811187	-2.07449568005484	6.49332970804459	4.52924871544827	21.7620923128663	15.3639706922949	-32.5704966008321	11.7053315563860	4.03101593092110	1.73850833926369	-20.6125635022933	15.1006787178167	14.1929561944557	11.0049696022916	10.4684343666547	-4.65084569675629	29.4546321793878	1.95890246946808	11.2639340133343];

figure(11)
plot(0:19, mulrssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(0:19, ones(1,20)*mean(mulrssa), 'k-', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-35 35])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('noised_a','-dtiff')

figure(12)
plot(0:19, mulrssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(0:19, ones(1,20)*mean(mulrssa), 'k-', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-35 35])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('noised_b','-dtiff')

% Sorting
textSize = 35;
sorta = sort(mulrssa);
sortb = sort(mulrssb);

figure(21)
plot(0:19, sorta, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(ones(1,8)*0, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*4.75, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*9.5, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*14.25, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*19, -35:10:35, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^A_' num2str(p)];
    text((p-1)*4.75+4.75/4, -25, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([-35 35])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('sort_seg_a','-dtiff')

figure(22)
plot(0:19, sortb, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(ones(1,8)*0, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*4.75, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*9.5, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*14.25, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*19, -35:10:35, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^B_' num2str(p)];
    text((p-1)*4.75+4.75/4, -25, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([-35 35])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('sort_seg_a','-dtiff')

% perm = randperm(20);
% [4 2 3 1]
perm = [16,17,18,19,20,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5];
% permInd = ind(perm);
shuf_rssa = sorta(perm);

figure(111)
plot(0:19, shuf_rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
plot(ones(1,8)*0, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*4.75, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*9.5, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*14.25, -35:10:35, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*19, -35:10:35, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^A_' num2str(p)];
    text((p-1)*4.75+4.75/4, -25, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([-35 35])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('shuff_seg_a','-dtiff')

% 
% figure(2)
% plot(0:19, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
% hold on
% plot(ones(1,4)*0, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*4.75, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*9.5, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*14.25, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*19, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% for p = 1:4
%     txt = ['p^A_' num2str(p)];
%     text((p-1)*4.75+4.75/4, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
%     hold on
% end
% xlim([xliml xlimr])
% ylim([-85 -50])
% set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
% set(gca,'xcolor', 'none', 'ycolor', 'none');
% print('perm_seg_a','-dtiff')
% 
% figure(21)
% plot(0:19, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
% hold on
% plot(ones(1,4)*0, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*4.75, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*9.5, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*14.25, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*19, -80:10:-50, 'b--', 'LineWidth', 2)
% hold on
% for p = 1:4
%     txt = ['p^B_' num2str(p)];
%     text((p-1)*4.75+4.75/4, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
%     hold on
% end
% xlim([xliml xlimr])
% ylim([-85 -50])
% set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
% set(gca,'xcolor', 'none', 'ycolor', 'none');
% print('perm_seg_b','-dtiff')
% 
% % Random Episode Permutation
% permEpi = [2 4 1 3];
% permEpiInd = [];
% 
% % for p = 1:4
% %     permEpiInd = [permEpiInd, ((permEpi(p)-1)*5+1):permEpi(p)*5];
% % end
% % permInd = permInd(permEpiInd);
% permInd = [132,129,123,120,134,128,124,135,125,127,121,118,131,126,122,130,116,117,119,133];
% rssa = A(permInd,1);
% % rssb = A(permInd,2);
% 
% rssa = (rssa-min(rssa))/(max(rssa)-min(rssa));
% rssb = (rssb-min(rssb))/(max(rssb)-min(rssb));
% 
% figure(3)
% plot(0:19, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
% hold on
% plot(ones(1,4)*0, -0.2:1.2/3:1, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*4.75, -0.2:1.2/3:1, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*9.5, -0.2:1.2/3:1, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*14.25, -0.2:1.2/3:1, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,4)*19, -0.2:1.2/3:1, 'b--', 'LineWidth', 2)
% hold on
% for p = 1:4
%     txt = ['p^A_' num2str(permEpi(p))];
%     text((p-1)*4.75+4.75/4, -0.2, txt, 'FontSize',textSize, 'FontWeight','bold');
%     hold on
% end
% xlim([xliml xlimr])
% ylim([-0.4 1.05])
% set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
% set(gca,'xcolor', 'none', 'ycolor', 'none');
% print('perm_a','-dtiff')