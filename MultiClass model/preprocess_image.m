function new_image = preprocess_image(filename)
% This function offers various preprocessing techniques for image
% manipulation.

% Read the Filename
img = imread(filename);

% Convert RGB images to grayscale
if ~ismatrix(img)
    img=rgb2gray(img); 
end

% 2D Gaussian filtering
 img = imgaussfilt(img,0.5);

% Median filtering
 img = medfilt2(img,[3,3]);

% Sharpening
 img = imsharpen(img);

% Contrast-limited adaptive histogram equalization (CLAHE)
 img = adapthisteq(img);

% Replicate the image 3 times to simulate an RGB image for compatibility
% with RESNET-50
new_image = cat(3,img,img,img);

end