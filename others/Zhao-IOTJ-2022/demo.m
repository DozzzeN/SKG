clear;clc
% load data_static_outdoor_1.mat
% load data_mobile_outdoor_1.mat
load data_static_indoor_1.mat
% load data_mobile_indoor_1.mat

% load csi_mobile_indoor_1_r.mat
% load csi_static_indoor_1_r.mat
% load csi_mobile_outdoor_r.mat
% load csi_static_outdoor_r.mat

% len = min(length(CSI_a1),length(CSI_b1));
% csi_a = CSI_a1(1:len);
% csi_b = CSI_b1(1:len);
% CSIa1 = csi_a;
% CSIb1 = csi_b;
% select = 2000:5000;
% CSIa1 = manipulationA;
% CSIb1 = CSI;
% CSIa1 = csi(:,1);
% CSIb1 = csi(:,2);
CSIa1 = A(:,1);
CSIb1 = A(:,2);
% CSIa1 = testdata(:,1);
% CSIb1 = testdata(:,2);
alpha = 0.2;

[testdata] = normalization(CSIa1,CSIb1);
[SBR,BMR,Entropy,a_list] = quantification(alpha,testdata);


