% Calculates the signal-to-noise ratio (SNR) in decibels (dB) of a color image
% 
% Created by Fernando Aristizabal
%

%enter your image's file path
I = single(imread('covid.png')); 

denoised_image = medfilt2(I);

%Calculations
mean_I = mean(mean(mean(I))); %calculates mean intensity value
N = numel(I); %calculates number of elements in the image
Signal = 255^2; %defines the max signal intensity
RMS_noise = sqrt(sum(sum(sum((I-mean_I).^2)))/N); %calculate root mean square noise of the image
SNR_dB = 10*log10(Signal/RMS_noise) %calculate and print SNR in dB

%Calculations
mean_I = mean(mean(mean(denoised_image))); %calculates mean intensity value
N = numel(denoised_image); %calculates number of elements in the image
Signal = 255^2; %defines the max signal intensity
RMS_noise = sqrt(sum(sum(sum((denoised_image-mean_I).^2)))/N); %calculate root mean square noise of the image
SNR_dB = 10*log10(Signal/RMS_noise) %calculate and print SNR in dB
