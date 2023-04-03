%% % % % % % % 观察两个不同CSI频域方差和皮尔森系数 % % % % %
clear;clc
test='1';
filename1 = strcat('csi1/CSI_',test,'.txt');
filename2 = strcat('csi2/CSI_',test,'.txt');
% filename1 = 'dynamicdata192.168.123.1/CSI_94.txt';
% filename2 = 'dynamicdata192.168.123.2/CSI_94.txt';
% filename1 = 'test/CSI_12.txt';
% filename2 = 'test/CSI_12-1.txt';
n=40;
%% 192.168.123.1 发送(相反)
[csi1, csi_time1, Fs1] = read_csi(filename1);
Fs1 = round(Fs1);
Time1 = 0:1/Fs1:((size(csi1,1)-1)/Fs1);
csi_time1 = Time1';
csi_value_tmp1 = abs(csi1);
csi_value1=[];

for i = 1: size(csi_value_tmp1,1)
    if(size(find(csi_value_tmp1(i,:)>10),2)<1)
        csi_value1 = [csi_value1;csi_value_tmp1(i,:)];   
    else
        csi_value1 = csi_value1;
    end
end
% csi_value1_sub = mean(csi_value1,1)';
% csi_value1_sub = csi_value1(n,:)';
% csi_value1_sub = mean(csi_value1,2);
csi_value1_sub = csi_value1(:,n);
% trans=[2.5265, 2.3036, 2.2019, 1.9798, 1.7899, 1.6519, 1.4779, 1.4611, 1.2963, 1.1957, 1.005, 0.95111, 0.77538, 0.76616, 0.65004, 0.68841, 0.58078, 0.54834, 0.52639, 0.51929, 0.49342, 0.47318, 0.43628, 0.40297, 0.38522, 0.35897,      0.34185, 0.36685, 0.38376, 0.41548, 0.45061, 0.46989, 0.49668, 0.50347, 0.52446, 0.55549, 0.65843, 0.63402, 0.74727, 0.75627, 0.92767, 0.98025, 1.1571, 1.2578, 1.3303, 1.4479, 1.6507, 1.7969, 1.9981, 2.1335, 2.2413, 2.4775]';
% rec=[1.633, 1.8282, 1.6969, 1.7564, 1.6866, 1.559, 1.5857, 1.5258, 1.5826, 1.414, 1.5096, 1.6692, 1.554, 1.501, 1.7082, 1.6545, 1.32, 1.2174, 1.165, 1.0146, 0.79747, 0.73912, 0.71853, 0.66199, 0.65902, 0.50128,     0.49504, 0.65082, 0.65376, 0.7096, 0.72992, 0.78755, 1.0156, 1.1661, 1.2186, 1.3212, 1.6561, 1.7811, 1.565, 1.6203, 1.7404, 1.574, 1.4236, 1.5932, 1.5361, 1.5964, 1.5695, 1.7347, 1.8065, 1.7453, 1.8803, 1.6796]';
% csi_value1_sub = csi_value1_sub.*trans;

len1 = size(csi_value1_sub);
mean1 = mean(csi_value1_sub);
var_f1 = sum((csi_value1_sub-mean1*ones(len1(1),1)).*(csi_value1_sub-mean1*ones(len1(1),1)))/len1(1);
var1 = var(csi_value1_sub);
std1 = std(csi_value1_sub);

%% 192.168.123.2 发送(双峰)
[csi2, csi_time2, Fs2] = read_csi(filename2);
Fs2 = round(Fs2);
Time2 = 0:1/Fs2:((size(csi2,1)-1)/Fs2);
csi_time2 = Time2';
csi_value_tmp2 = abs(csi2);

csi_value2=[];
for i = 1: size(csi_value_tmp2,1)
%     if(size(find(csi_value_tmp2(i,:)>10),2)>1 && size(find(csi_value_tmp2(i,:)<2),2)<40)
    if(size(find(csi_value_tmp2(i,:)>10),2)<1)
        csi_value2 = [csi_value2;csi_value_tmp2(i,:)]; 
    else
        csi_value2 = csi_value2;  
    end
end
% csi_value2_sub = mean(csi_value2,1)';
% csi_value2_sub = csi_value2(n,:)';
% csi_value2_sub = mean(csi_value2,2);
csi_value2_sub = csi_value2(:,n);
% trans=[2.5265, 2.3036, 2.2019, 1.9798, 1.7899, 1.6519, 1.4779, 1.4611, 1.2963, 1.1957, 1.005, 0.95111, 0.77538, 0.76616, 0.65004, 0.68841, 0.58078, 0.54834, 0.52639, 0.51929, 0.49342, 0.47318, 0.43628, 0.40297, 0.38522, 0.35897,      0.34185, 0.36685, 0.38376, 0.41548, 0.45061, 0.46989, 0.49668, 0.50347, 0.52446, 0.55549, 0.65843, 0.63402, 0.74727, 0.75627, 0.92767, 0.98025, 1.1571, 1.2578, 1.3303, 1.4479, 1.6507, 1.7969, 1.9981, 2.1335, 2.2413, 2.4775]';
% rec=[1.633, 1.8282, 1.6969, 1.7564, 1.6866, 1.559, 1.5857, 1.5258, 1.5826, 1.414, 1.5096, 1.6692, 1.554, 1.501, 1.7082, 1.6545, 1.32, 1.2174, 1.165, 1.0146, 0.79747, 0.73912, 0.71853, 0.66199, 0.65902, 0.50128,     0.49504, 0.65082, 0.65376, 0.7096, 0.72992, 0.78755, 1.0156, 1.1661, 1.2186, 1.3212, 1.6561, 1.7811, 1.565, 1.6203, 1.7404, 1.574, 1.4236, 1.5932, 1.5361, 1.5964, 1.5695, 1.7347, 1.8065, 1.7453, 1.8803, 1.6796]';
% csi_value2_sub = csi_value2_sub.*rec;


len2 = size(csi_value2_sub);
mean2 = mean(csi_value2_sub);
var_f2 = sum((csi_value2_sub-mean2*ones(len2(1),1)).*(csi_value2_sub-mean2*ones(len2(1),1)))/len2(1);
var2 = var(csi_value2_sub);
std2 = std(csi_value2_sub);


%% pearson相关系数
len = min(len1(1,1),len2(1,1));
% len=1000;
% p_csi1 = csi_value1_sub(192+316:len+191+316,1);
% p_csi2 = csi_value2_sub(192+25+16+316:len+191+25+16+316,1);
p_csi1 = csi_value1_sub(:,1)-min(csi_value1_sub(:,1));
p_csi2 = csi_value2_sub(:,1)-min(csi_value2_sub(:,1));
p_mean1 = mean(p_csi1); p_std1 = std(p_csi1);
p_mean2 = mean(p_csi2); p_std2 = std(p_csi2);


% conv = mean((p_csi1-p_mean1*ones(52,1)).*(p_csi2-p_mean2*ones(52,1)));
conv = mean((p_csi1(1:len,1)-p_mean1*ones(len,1)).*(p_csi2(1:len,1)-p_mean2*ones(len,1)));
p = conv/p_std1/p_std2;


%% 输出方差、皮尔森系数
var1
var2
p

%% 画图：画出两个图并且画出某一条载波
img1=csi_value1;
img1_sub=p_csi1;
figure(1)
hold on
surf(1:size(img1,1), 1:size(img1,2), img1', 'EdgeColor','none');
colormap jet
colorbar
caxis([0 2]);
ylim([1 size(img1,2)]);
xlabel('Time(seconds)')
ylabel('Frequency')
grid off
set(gca, 'FontSize', 10);

img2=csi_value2;
img2_sub=p_csi2;
figure(2)
hold on
surf(1:size(img2,1), 1:size(img2,2), img2', 'EdgeColor','none');
colormap jet
colorbar
% caxis([0 11.6]);
ylim([1 size(img2,2)]);
xlabel('Time(seconds)')
ylabel('Frequency')
grid off
set(gca, 'FontSize', 10);

figure(3)
hold on
plot(1:size(img1_sub,1),img1_sub,1:size(img2_sub,1),img2_sub)
ylabel('Amplitude')
xlabel('Subcarrier')


%% 
len = min(length(p_csi1),length(p_csi2))
testdata=[];
testdata(:,1)= p_csi1(1:len);
testdata(:,2)= p_csi2(1:len);
savename = strcat('testdata_',test,'.mat');
save(savename,'testdata');




