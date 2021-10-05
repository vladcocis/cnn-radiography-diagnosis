% Calculate signal-to-noise ratio (SNR) in decibels (dB)

% Load image
original_image = imread('covid.png');

%% SNR of original image

% Convert to single precision
image = single(original_image); 

% Calculate the mean intensity
mean_intensity = mean(mean(mean(image)));

% Calculate number of elements
elements = numel(image); 

% Set maximum signal intensity
max = 255^2;

% Calculate root of mean square noise of the image
noise = sqrt(sum(sum(sum((image-mean_intensity).^2)))/elements); 

% Calculate and display SNR in dB
SNR = 10*log10(max/noise) 

%% SNR of filtered image

% Convert to single precision
filtered_image = single(original_image); 

% Use same image processing techniques as preprocessing function:
% Gaussian filter
filtered_image = imgaussfilt(filtered_image,0.5);

% Median filter
filtered_image = medfilt2(filtered_image,[3,3]);

% Sharpening
filtered_image = imsharpen(filtered_image);

% Contrast-Limited Adaptive Histogram Equalisation (CLAHE)
filtered_image = adapthisteq(filtered_image);

% Calculate the mean intensity
mean_intensity = mean(mean(mean(filtered_image)));

% Calculate number of elements
elements = numel(filtered_image); 

% Set maximum signal intensity
max = 255^2;

% Calculate root of mean square noise of the image
noise = sqrt(sum(sum(sum((filtered_image-mean_intensity).^2)))/elements); 

% Calculate and display SNR in dB
SNR_filtered = 10*log10(max/noise) 

%% Display the two images side by side
% Images can only be shown without converting to single precision
% so filtering is done again on original image.

% Gaussian filter
processed_image = imgaussfilt(original_image,0.5);

% Median filter
processed_image = medfilt2(processed_image,[3,3]);

% Sharpening
processed_image = imsharpen(processed_image);

% Contrast-Limited Adaptive Histogram Equalisation (CLAHE)
processed_image = adapthisteq(processed_image);

% Display images
figure
montage({original_image,processed_image})
title(['Original Image SNR: ',num2str(SNR),'     Vs.  Filtered Image SNR: ',num2str(SNR_filtered)])