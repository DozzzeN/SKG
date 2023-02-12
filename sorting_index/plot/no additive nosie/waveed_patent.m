clear all; close all; clc

load data_mobile_indoor_2.mat

% Original RSS Sequence
ind = 116:135;
rssa = A(ind,1);
rssb = A(ind,2);

insertind = 236:240;
rssinsert = A(insertind,1);

xliml = -1;
xlimr = 20;

figure(1)
plot(0:19, rssa, 'ko-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(0:19, ones(1,20)*-65, 'k-', 'LineWidth',2)
ylim([-80 -50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('ori_a','-dtiff')

figure(11)
plot(0:19, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(0:19, ones(1,20)*-65, 'k-', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-80 -50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('ori_b','-dtiff')

% Permutation
textSize = 35;
% perm = randperm(20);
perm = [6,3,16,11,7,17,14,8,5,19,15,1,2,4,18,13,9,20,10,12];
permInd = ind(perm);
rssa = A(permInd,1);
rssb = A(permInd,2);

figure(2)
plot(0:19, rssa, 'ko-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^A_' num2str(p)];
    text((p-1)*4.75+4.75/4, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([-85 -50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('perm_seg_a','-dtiff')

figure(21)
plot(0:19, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -80:10:-50, 'k--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^B_' num2str(p)];
    text((p-1)*4.75+4.75/4, -80, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
xlim([xliml xlimr])
ylim([-85 -50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('perm_seg_b','-dtiff')

% Random Episode Editing
permEpi = [1 2 4 3];

% insert the 1th episode and delete the 4th episode
rsse = [1:20].';
for e = 1:5
    rsse(e) = rssinsert(e);
end

for e = 1:10
    rsse(5+e) = rssa(e);
end

for e = 15:20
    rsse(e) = rssa(e);
end 

rsse_all = [1:25].';
for e = 1:5
    rsse_all(e) = rssinsert(e);
end

for e = 1:20
    rsse_all(5+e) = rssa(e);
end

rssa = (rssa-min(rssa))/(max(rssa)-min(rssa));
rssb = (rssb-min(rssb))/(max(rssb)-min(rssb));
rsse = (rsse-min(rsse))/(max(rsse)-min(rsse));
rsse_all = (rsse_all-min(rsse_all))/(max(rsse_all)-min(rsse_all));

figure(3)
plot(0:24, rsse_all, 'ko-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*23.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
text((1-1)*4.75+4.75/5, -0.2, ['p^A_i'], 'FontSize',textSize, 'FontWeight','bold');
text((2-1)*4.75+4.75/5, -0.2, ['p^A_1'], 'FontSize',textSize, 'FontWeight','bold');
text((3-1)*4.75+4.75/5, -0.2, ['p^A_2'], 'FontSize',textSize, 'FontWeight','bold');
text((4-1)*4.75+4.75/5, -0.2, ['p^A_3'], 'FontSize',textSize, 'FontWeight','bold');
text((5-1)*4.75+4.75/5, -0.2, ['p^A_4'], 'FontSize',textSize, 'FontWeight','bold');
xlim([xliml xlimr+5])
ylim([-0.4 1.05])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('edit_a','-dtiff')

figure(4)
plot(0:24, rsse_all, 'ko-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*23.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
text((1-1)*4.75+4.75/5, -0.2, ['p^A_i'], 'FontSize',textSize, 'FontWeight','bold');
text((2-1)*4.75+4.75/5, -0.2, ['p^A_1'], 'FontSize',textSize, 'FontWeight','bold');
text((3-1)*4.75+4.75/5, -0.2, ['p^A_2'], 'FontSize',textSize, 'FontWeight','bold');
text((4-1)*4.75+4.75/5, -0.2, ['p^A_3'], 'FontSize',textSize, 'FontWeight','bold');
text((5-1)*4.75+4.75/5, -0.2, ['p^A_4'], 'FontSize',textSize, 'FontWeight','bold');
% patch([0 4.75 4.75 0], [1.1 1.1 -1 -1], [157 195 230]/255,'facealpha', 0.5)
% patch([4.75*3 4.75*4 4.75*4 4.75*3], [1.1 1.1 -1 -1], [169 209 142]/255, 'facealpha', 0.5)
hold on
xlim([xliml xlimr+5])
ylim([-0.4 1.05])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('edit_a_remain','-dtiff')

figure(41)
plot(0:19, rssa, 'ko-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^A_' num2str(p)];
    text((p-1)*4.75+4.75/4, -0.2, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
% patch([4.75*2 4.75*3 4.75*3 4.75*2], [1.1 1.1 -1 -1], [0.5 0.5 0.5], 'facealpha', 0.5)
hold on
xlim([xliml xlimr])
ylim([-0.4 1.05])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('a_remain','-dtiff')

figure(42)
plot(0:19, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['p^B_' num2str(p)];
    text((p-1)*4.75+4.75/4, -0.2, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
% patch([4.75*2 4.75*3 4.75*3 4.75*2], [1.1 1.1 -1 -1], [169 209 142]/255, 'facealpha', 0.5)
hold on
xlim([xliml xlimr])
ylim([-0.4 1.1])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('b_remain','-dtiff')

figure(43)
plot(0:24, rsse_all, 'ko-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
plot(ones(1,4)*0, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*4.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*9.5, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*14.25, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*19, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
plot(ones(1,4)*23.75, -0.2:1.2/3:1, 'k--', 'LineWidth', 2)
hold on
text((1-1)*4.75+4.75/5, -0.2, ['p^A_i'], 'FontSize',textSize, 'FontWeight','bold');
text((2-1)*4.75+4.75/5, -0.2, ['p^A_1'], 'FontSize',textSize, 'FontWeight','bold');
text((3-1)*4.75+4.75/5, -0.2, ['p^A_2'], 'FontSize',textSize, 'FontWeight','bold');
text((4-1)*4.75+4.75/5, -0.2, ['p^A_3'], 'FontSize',textSize, 'FontWeight','bold');
text((5-1)*4.75+4.75/5, -0.2, ['p^A_4'], 'FontSize',textSize, 'FontWeight','bold');
% patch([4.75*3 4.75*4 4.75*4 4.75*3], [1.1 1.1 -1 -1], [0.5 0.5 0.5], 'facealpha', 0.5)
hold on
xlim([xliml xlimr+5])
ylim([-0.4 1.05])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('edit_e_remain','-dtiff')