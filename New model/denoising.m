new_image = imread('covid.png');

noisy_image = imnoise(new_image,'salt & pepper');

[peaksnr, snr] = psnr(noisy_image, new_image);

fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
fprintf('\n The SNR value is %0.4f \n', snr);


denoised_image = adapthisteq(denoised_image);

[peaksnr, snr] = psnr(noisy_image, denoised_image);

fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
fprintf('\n The SNR value is %0.4f \n', snr);

montage({new_image,denoised_image})
title('Original Image (Left) Vs. Filtered Image (Right)')