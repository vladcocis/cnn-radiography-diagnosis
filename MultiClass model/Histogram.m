%% Display histogram of image before and after preprocessing

% Load image
original_image = imread('covid.png');

% Use same image processing techniques as preprocessing function:
% Gaussian filter
filtered_image = imgaussfilt(original_image,0.5);

% Median filter
filtered_image = medfilt2(filtered_image,[3,3]);

% Sharpening
filtered_image = imsharpen(filtered_image);

% Contrast-Limited Adaptive Histogram Equalisation (CLAHE)
filtered_image = adapthisteq(filtered_image);

% Display histogram of original image
figure;
imhist(original_image);
title("Original Image");

% Display histogram of filtered image
figure;
imhist(filtered_image);
title("CLAHE");

% Display original image
figure;
imshow(original_image);
title("Original Image");

% Display filtered image
figure;
imshow(filtered_image);
title("Filtered Image");