% First load the electrical signal and select a segment from it. Plot the segment.
load leleccum
indx = 2000:3450;
x = leleccum(indx);
figure
plot(indx,x)
axis tight
title('Original Signal')

% Denoise the signal using the db3 wavelet and a three-level wavelet 
% decomposition and soft fixed form thresholding. To deal with the 
% non-white noise, use level-dependent noise size estimation. 
% Compare with the original signal.

xd = wdenoise(x,3,'Wavelet','db3',...
    'DenoisingMethod','UniversalThreshold',...
    'ThresholdRule','Soft',...
    'NoiseEstimate','LevelDependent');
figure
subplot(2,1,1)
plot(indx,x)
axis tight
title('Original Signal')
subplot(2,1,2)
plot(indx,xd)
axis tight
title('Denoised Signal')