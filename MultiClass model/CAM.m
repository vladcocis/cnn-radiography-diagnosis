%% Display class activation mappings using GradCAM
% this needs to be run after training the model using the classifier.m file

% load the CNN
net = netTransfer;

% load the image
im = imread("test.jpg");

% resize image to fit ResNet50
img = imresize(im,[224 224]);

% classify image using trained model
[classfn,score] = classify(net,img)

% display image and classification result
figure;
imshow(img);
title(sprintf("%s (%.2f)", classfn, score(classfn)));

% create heatmap of important areas using gradCAM
map = gradCAM(net,img,classfn);

% display heatmap
figure;
imshow(img);
hold on;
imagesc(map,'AlphaData',0.5);
colormap jet
hold off;
title("Grad-CAM");