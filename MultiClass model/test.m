sample_image_folder = fullfile(matlabroot,'toolbox/images/imdata');
[filename,user_canceled] = imgetfile('InitialPath',sample_image_folder);
img = imread(filename);

% Resize image to 224x224
img_resized = imresize(img,[224 224]);

% Classify image using trained CNN.
[class, score]=classify(netTransfer,img_resized);

% Display image and classification result
figure
imshow(img_resized)
title([ 'Prediction:' char(string(class)),', ','chance=',num2str(max(score)*100),'%, ','Class:', char(string(testing_set.Labels(1)))])