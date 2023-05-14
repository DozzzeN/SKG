close all; clc
textSize = 16;
linewid = 1;

figure(1)
% f1出现bug
nums=99;
[f1,x1]=ksdensity(legiCorr(1:nums));
[f2,x2]=ksdensity(randomCorr(1:nums));
[f3,x3]=ksdensity(inferCorr(1:nums));
[f4,x4]=ksdensity(imitCorr(1:nums));
[f5,x5]=ksdensity(stalkCorr(1:nums));

f1=f1/length(legiCorr);
f2=f2/length(randomCorr);
f3=f3/length(inferCorr);
f4=f4/length(imitCorr);
f5=f5/length(stalkCorr);

y_max=1; %纵轴刻度最大值
y_interval=0.1;
y_break_start=0.1; % 截断的开始值
y_break_end=0.9; % 截断的结束值

adjust_value=0.4*y_interval; %微调截断处y坐标
uptate_num=y_break_end-y_break_start-y_interval; %最高处曲线向下平移大小
 
% 超过截断结束位置的那些曲线统统向下平移uptate_num个长度
for i=1:length(legiCorr(:, 1))
    if legiCorr(i, :)>y_break_end
        legiCorrb(i, :)=legiCorr(i, :)-uptate_num;
    end
end

legiCorrb(1)=0;  % 保证能连成一条曲线
x1b=ones([1, length(legiCorrb)]);

lineWidth=1;
h1=plot(x1b, legiCorrb, 'o-', 'LineWidth',lineWidth);
hold on
h2=plot(x2, f2, '*--', 'LineWidth',lineWidth);
hold on
h3=plot(x3, f3, 'd:', 'LineWidth',lineWidth);
hold on
h4=plot(x4, f4, '+-.', 'LineWidth',lineWidth);
hold on
h5=plot(x5, f5, 'x-', 'LineWidth',lineWidth);
hold on
ax = gca;
ax.YAxis.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.LineWidth = 1;

% 这样写legend的style才不会错乱
legend([h1 h2 h3 h4 h5], {'Original', 'Random Guess Attack',...
        'Inference Attack', 'Imitation Attack', 'Stalking Attack'}, ...
        'FontSize', textSize, 'FontWeight','bold', 'Location','northwest')
xlabel('\rho(RO,RO''),\rho(RO,RO^E)', 'FontSize', 15, 'FontWeight','bold')
ylabel('PDF', 'FontSize',15, 'FontWeight','bold')

ylim([0 0.4]);
% 纵坐标截断设置
ylimit=get(gca,'ylim');
location_Y=(y_break_start+adjust_value-ylimit(1))/diff(ylimit);
t1=text(0, location_Y,'//','sc','BackgroundColor','w','margin',eps, 'fontsize',14);
set(t1,'rotation',90);
t2=text(1, location_Y,'//','sc','BackgroundColor','w','margin',eps, 'fontsize',14);
set(t2,'rotation',90);

% 重新定义纵坐标刻度
ytick=0:y_interval:y_max;
set(gca,'ytick',ytick);
ytick(ytick>y_break_start+eps)=ytick(ytick>y_break_start+eps)+uptate_num;
for i=1:length(ytick)
   yticklabel{i}=sprintf('%.2g',ytick(i));
end
set(gca,'yTickLabel', yticklabel, 'FontSize', 14); %修改坐标名称、字体
grid on

print('corrkeys','-depsc')