% First set a signal-to-noise ratio (SNR) and set a random seed.
sqrt_snr = 4;
init = 2055615866;


% Generate an original signal xref and a noisy version x 
% by adding standard Gaussian white noise. Plot both signals.
% test signal name=1
% plot 2^11 linearly spaced points from 0 to 1
% x corrupted by AGWN N(0,1) and has a SNR of sqrtsnr^2
% init is the seed in generating AGWN
[xref,x] = wnoise(1,11,sqrt_snr,init);
subplot(2,1,1)
plot(xref)
axis tight
title('Original Signal')
subplot(2,1,2)
plot(x)
axis tight
title('Noisy Signal')

% Denoise the noisy signal using soft heuristic SURE thresholding 
% on detail coefficients obtained from the wavelet decomposition 
% of x using the sym8 wavelet. Use the default settings of wdenoise 
% for the remaining parameters. Compare with the original signal.
xd = wdenoise(x,'Wavelet','sym8','DenoisingMethod','SURE','ThresholdRule','Soft');
figure
subplot(2,1,1)
plot(xref)
axis tight
title('Original Signal')
subplot(2,1,2)
plot(xd)
axis tight
title('Denoised Signal')