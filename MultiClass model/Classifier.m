clc;
clear;
close all;

%% Load the Dataset
dataset='D:\Uni Work\Project\datasetSmall';
 
% Store all images in an Image Datastore
image_datastore=imageDatastore(dataset, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% Display some Images
% Calculate total number of images
number_of_images=length(image_datastore.Labels);

% set number of images to display
x=6;

% Display x random images
image_ids=randperm(number_of_images,x);
figure;
for i=1:length(image_ids)
    
    subplot(3,2,i);
    imshow(imread(image_datastore.Files{image_ids(i)}));
    title((image_datastore.Labels(image_ids(i))))
    
end

%% Preprocessing on dataset
% preprocessing is done in separate function
image_datastore.ReadFcn = @(filename)preprocess_image(filename);

%% Split Data for Training and Testing
% Data is using 80:20 split
[training_set,testing_set] = splitEachLabel(image_datastore,.8,'randomized');

%% Image augmentation
% create augmenter with parameters
aug = imageDataAugmenter( ...
        'RandRotation',[-3 3],...
        'RandXReflection',1,...
        'RandYReflection',1);
    
% apply augmenter and set output size to 224x224
augmented_images = augmentedImageDatastore([224 224],training_set,'DataAugmentation',aug);

%% Load ResNet50
% Get pretrained ResNet50 from MATLAB Deep Learning toolbox
% Store the net into a layer graph to visualise and modify layers
lgraph = layerGraph(resnet50);

% Visualize names of all layers
network_layers = lgraph.Layers

% Get number of output classes in ResNet50. 
% Original ResNet50 has 1000 output classes. 
% This needs to be changed for multi-class classification of X-Ray scans.
lgraph.Layers(175).OutputSize

%% Count number of output classes
output_classes = numel(categories(training_set.Labels))

%% Replace last layers due to changed number of output classes
% Create a new fully-connected layer
new_fc = fullyConnectedLayer(output_classes, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

% Replace fully-connected layer in ResNet50
lgraph = replaceLayer(lgraph,'fc1000',new_fc);

% Create new SoftMax layer
new_softmax = softmaxLayer('Name','new_softmax');

% Replace SoftMax layer in ResNet-50
lgraph = replaceLayer(lgraph,'fc1000_softmax',new_softmax);

% Create a new classification layer
% Output classes of the layer are automatically set at training time.
new_output = classificationLayer('Name','new_output');

% Replace classification layer in ResNet-50
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',new_output);

%% Training options
options = trainingOptions('adam',...
        'MiniBatchSize',20,...
        'MaxEpochs',5,...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'ExecutionEnvironment','gpu',...
        'Plots','training-progress');
    
%% Training
netTransfer = trainNetwork(augmented_images,lgraph,options);

%% Testing
% Resize all images using output size 224x224
augtestimds = augmentedImageDatastore([224 224],testing_set);

% Compute predictions
predicted_labels = classify(netTransfer,augtestimds);

% Calculate prediction accuracy
sum(predicted_labels==testing_set.Labels)/numel(predicted_labels)*100

% Create Labels
actual_labels=testing_set.Labels;

% Display Confusion Matrix
figure;
plotconfusion(actual_labels,predicted_labels)
title('Confusion Matrix: Res-Net50');

%% Classify a new image
% Get a new image
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
title([ 'Prediction:' char(string(class)),', ','chance=',num2str(max(score)*100),'%, '])