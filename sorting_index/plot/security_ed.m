clear; close all; clc
textSize = 14;
linewid = 1;

points128 = [];
points96 = [];
points64 = [];
points48 = [];
points32 = [];
lowers = [];

for i=1:129
    lowers(i) = 2^(i-1);
end

mp = 128;
for i=1:129
    tmp = 0;
    for mr=1:i
        if (mr-1<=mp) && (mr-1<=i-1)
            tmp = tmp + nchoosek(i-1,mr-1) * nchoosek(mp,mr-1);
        end
    end
    points128(i) = tmp;
end

mp = 96;
for i=1:129
    tmp = 0;
    for mr=1:i
        if (mr-1<=mp) && (mr-1<=i-1)
            tmp = tmp + nchoosek(i-1,mr-1) * nchoosek(mp,mr-1);
        end
    end
    points96(i) = tmp;
end

mp = 64;
for i=1:129
    tmp = 0;
    for mr=1:i
        if (mr-1<=mp) && (mr-1<=i-1)
            tmp = tmp + nchoosek(i-1,mr-1) * nchoosek(mp,mr-1);
        end
    end
    points64(i) = tmp;
end

mp = 48;
for i=1:129
    tmp = 0;
    for mr=1:i
        if (mr-1<=mp) && (mr-1<=i-1)
            tmp = tmp + nchoosek(i-1,mr-1) * nchoosek(mp,mr-1);
        end
    end
    points48(i) = tmp;
end

mp = 32;
for i=1:129
    tmp = 0;
    for mr=1:i
        if (mr-1<=mp) && (mr-1<=i-1)
            tmp = tmp + nchoosek(i-1,mr-1) * nchoosek(mp,mr-1);
        end
    end
    points32(i) = tmp;
end

figure1 = figure;
plot([1:129], points128, 'b--', 'LineWidth',2)
hold on
plot([1:129], points96, 'm-', 'LineWidth',2)
hold on
plot([1:129], points64, 'c:', 'LineWidth',2)
hold on
plot([1:129], points48, 'g-.', 'LineWidth',2)
hold on
plot([1:129], points32, 'k--', 'LineWidth',2)
hold on
plot([1:129], lowers, 'r-', 'LineWidth',2)
hold on
plot(104,1.3502058579275469e+31,'ro','LineWidth',2);
hold on;

ax = gca;
ax.YAxis.FontSize = 13;
ax.XAxis.FontSize = 13;
ax.YAxis.FontWeight = 'bold';
ax.XAxis.FontWeight = 'bold';
ax.YScale='log';
ax.YTick=[0, 1e+20, 1e+40, 1e+60, 1e+80, 1e+100, 1e+120];

ylim([0, 1e+120]);
legend('M_{guess}^{SA}, M_p=128','M_{guess}^{SA}, M_p=96',...
    'M_{guess}^{SA}, M_p=64', 'M_{guess}^{SA}, M_p=48',...
    'M_{guess}^{SA}, M_p=32', '2^M', ...
    'Position',[0.151190476190476 0.630952380952381 0.653571428571428 0.278571428571429],...
    'NumColumns',2, 'FontSize',textSize,...
    'FontWeight','bold', 'Location','northwest')
xlabel('M', 'FontSize', 15, 'FontWeight','bold')
ylabel('The number of guesses', 'FontSize',15, 'FontWeight','bold')
grid on

% 创建 arrow
annotation(figure1,'arrow',[0.653571428571429 0.704761904761904],...
    [0.233333333333333 0.325396825396826],'LineWidth',1);

% 创建 textbox
annotation(figure1,'textbox',...
    [0.516666666666666 0.142857142857143 0.273809523809524 0.0968253968253976],...
    'String',{'(104, 1.35e+31)'},...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',14,...
    'FitBoxToText','off');

print('nAllPlot_ed','-depsc')