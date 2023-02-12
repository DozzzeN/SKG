clear all; close all; clc

load data_mobile_indoor_2.mat

noteSize = 35;
textSize = 35;

% Original RSS Sequence
ind = 2145:2145+19;
% rssa = A(ind,1);
% rssb = A(ind,2);

rssa = [-64 -63 -60 -68 -67 -70 -77 -61];
rssb = [-63 -64 -61 -68 -67 -70 -77 -60];

rssa = abs(rssa);
rssb = abs(rssb);

xliml = -1;
xlimr = 8;

figure(1)
bar(0:7, rssa, 'r', 'grouped')
% plot(0:7, rssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
% plot(0:7, ones(1,8)*mean(rssa), 'k--', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-5 100])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% for i=1:length(rssa)
%     if i == 4
%         text(i-1,rssa(i)-2,num2str(rssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i == 5
%         text(i-1,rssa(i)-2,num2str(rssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i == 7
%         text(i-1,rssa(i)-2,num2str(rssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     else
%         text(i-1,rssa(i)+0.5,num2str(rssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     end
% end

print('ori_a','-dtiff')

figure(2)
bar(0:7, rssb, 'k', 'grouped');
% plot(0:7, rssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
% plot(0:7, ones(1,8)*mean(rssa), 'k--', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-5 100])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% for i=1:length(rssb)
%     if i == 4
%         text(i-1,rssb(i)-2,num2str(rssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i == 5
%         text(i-1,rssb(i)-2,num2str(rssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i == 7
%         text(i-1,rssb(i)-2,num2str(rssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     else
%         text(i-1,rssb(i)+0.5,num2str(rssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     end
% end

print('ori_b','-dtiff')

% Add Noise
mulnoise = unifrnd(0,3,length(rssa),length(rssa));
% mulrssa = (rssa - mean(rssa)) * mulnoise;
% mulrssb = (rssb - mean(rssb)) * mulnoise;

mulrssa = [9.02 -17.45 7.04 -1.82 -35.41 6.73 12.46 9.72];
mulrssb = [9.01 -17.52 7.55 -3.19 -34.91 5.98 10.92 12.99];

mulrssa = abs(mulrssa);
mulrssb = abs(mulrssb);

sorta = sort(mulrssa);
sortb = sort(mulrssb);

figure(11)
bar(0:7, mulrssa, 'r', 'grouped')
% plot(0:7, mulrssa, 'ro-', 'MarkerFaceColor', 'r', 'LineWidth',5)
hold on
% plot(0:7, ones(1,8)*mean(mulrssa), 'k--', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-5 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% for i=1:length(mulrssa)
%     if i==2
%         text(i-1,mulrssa(i)-5,num2str(mulrssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i==5
%         text(i-1,mulrssa(i)-5,num2str(mulrssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i==4
%         text(i-1,mulrssa(i)+3,num2str(mulrssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     else
%         text(i-1,mulrssa(i)+1,num2str(mulrssa(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     end
% end
print('noised_a','-dtiff')

figure(12)
bar(0:7, mulrssb, 'k', 'grouped')
% plot(0:7, mulrssb, 'ks-', 'MarkerFaceColor', 'k', 'LineWidth',5)
hold on
% plot(0:7, ones(1,8)*mean(mulrssa), 'k--', 'LineWidth',2)
xlim([xliml xlimr])
ylim([-5 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

% for i=1:length(mulrssb)
%     if i==2
%         text(i-1,mulrssb(i)-5,num2str(mulrssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i==5
%         text(i-1,mulrssb(i)-5,num2str(mulrssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     elseif i==4
%         text(i-1,mulrssb(i)+3,num2str(mulrssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     else
%         text(i-1,mulrssb(i)+1,num2str(mulrssb(i), '%.2g'), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
%     end
% end
print('noised_b','-dtiff')

% Index
% sorta = [6 2 5 3 1 4 8 7];
% sortb = [6 2 5 3 1 4 7 8];
figure(21)
bar(0:7, mulrssa, 'r', 'grouped')
hold on
% range = -4:2:10;
% plot(ones(1,8)*-0.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*1.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*3.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*5.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*7.5, range, 'b--', 'LineWidth', 2)
% hold on
% for p = 1:4
%     txt = ['e^A_' num2str(p)];
%     text((p-1)*2, -2, txt, 'FontSize',textSize, 'FontWeight','bold');
%     hold on
% end

ori_ind = [1,2,3,4,5,6,7,8];
for i=1:length(ori_ind)
    %     text((i-1),sorta(i),strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),40,strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
xlim([xliml xlimr])
ylim([-5 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('ind_seg_a','-dtiff')

figure(22)
bar(0:7, mulrssb, 'k', 'grouped')
hold on
% range = -4:2:10;
% plot(ones(1,8)*-0.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*1.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*3.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*5.5, range, 'b--', 'LineWidth', 2)
% hold on
% plot(ones(1,8)*7.5, range, 'b--', 'LineWidth', 2)
% hold on
% for p = 1:4
%     txt = ['e^B_' num2str(p)];
%     text((p-1)*2, -2, txt, 'FontSize',textSize, 'FontWeight','bold');
%     hold on
% end

for i=1:length(sortb)
    %     text((i-1),sortb(i),strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),40,strcat(num2str(ori_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
xlim([xliml xlimr])
ylim([-5 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('ind_seg_b','-dtiff')

% sorting


figure(31)
bar(0:7, sorta, 'r', 'grouped')
hold on
range = -20:10:50;
plot(ones(1,8)*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*1.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*3.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*5.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*7.5, range, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['e^A_' num2str(p)];
    text((p-1)*2, -10, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end

sorta_ind = [4, 6, 3, 1, 8, 7, 2, 5];
for i=1:length(sorta)
    %     text((i-1),sorta(i),strcat(num2str(sorta_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),40,strcat(num2str(sorta_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
xlim([xliml xlimr])
ylim([-20 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('sort_seg_a','-dtiff')

figure(32)
bar(0:7, sortb, 'k', 'grouped')
hold on
range = -20:10:50;
plot(ones(1,8)*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*1.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*3.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*5.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*7.5, range, 'b--', 'LineWidth', 2)
hold on
for p = 1:4
    txt = ['e^B_' num2str(p)];
    text((p-1)*2, -10, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end

sortb_ind = [4, 6, 3, 1, 7, 8, 2, 5];
for i=1:length(sortb)
    %     text((i-1),sortb(i),strcat(num2str(sortb_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),40,strcat(num2str(sorta_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
xlim([xliml xlimr])
ylim([-20 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');
print('sort_seg_b','-dtiff')

% shuffling
shuf_rssa = [sorta(5),sorta(6),sorta(1),sorta(2),sorta(3),sorta(4),sorta(7),sorta(8)];

figure(111)
bar(0:7, shuf_rssa, 'r', 'grouped')
hold on
range = -20:10:50;
plot(ones(1,8)*-0.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*1.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*3.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*5.5, range, 'b--', 'LineWidth', 2)
hold on
plot(ones(1,8)*7.5, range, 'b--', 'LineWidth', 2)
hold on

% for p = 1:4
%     text((p-1)*2, -2, ['e^A_' num2str(shuf_ind(p))], 'FontSize',textSize, 'FontWeight','bold');
%     hold on
% end

shuf_ind = [8,7,4,6,3,1,2,5];
for i=1:length(shuf_rssa)
    %     text((i-1),shuf_rssa(i),strcat(num2str(shuf_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text((i-1),40,strcat(num2str(shuf_ind(i))), 'FontSize',noteSize, 'FontWeight','bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    hold on
end
xlim([xliml xlimr])
ylim([-20 50])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gca,'xcolor', 'none', 'ycolor', 'none');

perm=[3 1 2 4];
for p = 1:4
    txt = ['e^A_' num2str(perm(p))];
    text((p-1)*2, -10, txt, 'FontSize',textSize, 'FontWeight','bold');
    hold on
end
print('shuff_seg_a','-dtiff')