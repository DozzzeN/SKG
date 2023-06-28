file_name = 'data_static_indoor_1';
load(strcat(file_name,'.mat'));

a=A(1:500,1);
b=A(1:500,2);
subplot(221);
plot(a); title('Original A');
subplot(222);
plot(b); title('Original B');

thr=2;
a = wthresh(a,'s',thr);
b = wthresh(b,'s',thr);

% subplot(223);
% plot(a); title('Thr A');
% subplot(224);
% plot(b); title('Thr B');

pearson=corr(a,b,'type','pearson');
spearman= corr(a,b,'type','Spearman');

wavelet='sym8';
ThresholdRule='Soft';
DenoisingMethod='UniversalThreshold';
NoiseEstimate='LevelIndependent';
a = wdenoise(a,'Wavelet',wavelet,'DenoisingMethod',DenoisingMethod,...
    'ThresholdRule',ThresholdRule,'NoiseEstimate',NoiseEstimate);
b = wdenoise(b,'Wavelet',wavelet,'DenoisingMethod',DenoisingMethod,...
    'ThresholdRule',ThresholdRule,'NoiseEstimate',NoiseEstimate);

subplot(223);
plot(a); title('Denoised A');
subplot(224);
plot(b); title('Denoised B');

pearsond=corr(a,b,'type','pearson');
spearmand= corr(a,b,'type','Spearman');

span=15;
method='moving';
a = smooth(a,span,method);
b = smooth(b,span,method);
pearsondm=corr(a,b,'type','pearson');
spearmandm= corr(a,b,'type','Spearman');

% save(strcat(file_name,'_m.mat'),'A');
% fprintf(strcat(file_name,' done'));