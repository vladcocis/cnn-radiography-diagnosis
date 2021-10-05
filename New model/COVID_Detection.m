%% Clear workspace
clear, clc, close all;

%% Load the Dataset
dataset='D:\Uni Work\Project\datasetLarge';
 
% Store all images in an Image Datastore
image_datastore=imageDatastore(dataset, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% Count number of samples for every class
total_images = countEachLabel(image_datastore) 

%% Display some Images
% Calculate total number of images
number_of_images=length(image_datastore.Labels);

% set number of images to display
x=6;

% Display x random images
image_ids=randperm(number_of_images,x);
figure;
for i=1:length(image_ids)
    
    subplot(2,3,i);
    imshow(imread(image_datastore.Files{image_ids(i)}));
    title((image_datastore.Labels(image_ids(i))))
    
end

%% Preprocessing on dataset
% preprocessing is done in separate function
image_datastore.ReadFcn = @(filename)preprocess_image(filename);

%% Split Data for Training and Testing
% Data is using 80:20 split
[training_set,testing_set] = splitEachLabel(image_datastore,.5,'randomized');

%% Image augmentation
% create augmenter with parameters
augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    
% apply augmenter and set output size to 224x224
augmented_images = augmentedImageDatastore([224 224],training_set,'DataAugmentation',augmenter);

%% Load ResNet-50
% Get pretrained ResNet-50 from MATLAB Deep Learning toolbox
% Store the net into a layer graph to visualise and modify layers
lgraph = layerGraph(resnet50);

% Visualize all layers
network_layers = lgraph.Layers

% Get number of output classes in ResNet-50. 
% Original ResNet-50 has 1000 output classes. 
% This needs to be changed for binary classification of X-Ray scans.
lgraph.Layers(175).OutputSize

%% Count number of output classes
% There will be 2 output classes: covid and normal
output_classes = numel(categories(training_set.Labels))

%% Replace last layers due to changed number of output classes
% Create a new fully-connected layer
new_fc = fullyConnectedLayer(output_classes, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

% Replace fully-connected layer in ResNet-50
lgraph = replaceLayer(lgraph,'fc1000',new_fc);

% Create new SoftMax layer
new_softmax = softmaxLayer('Name','new_softmax');

% Replace SoftMax layer in ResNet-50
lgraph = replaceLayer(lgraph,'fc1000_softmax',new_softmax);

% Create a new classification layer
% Output classes of the layer are automatically set at training time.
new_classoutput = classificationLayer('Name','new_classoutput');

% Replace classification layer in ResNet-50
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',new_classoutput);

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

%% Visualise the activations of a layer
% Get and image from the testing set
img = preprocess_image(testing_set.Files{5});

% Resize image to 224x224
img_resized = imresize(img,[224 224]);

% Specify target layer for activation visualisation.
% Features like edges are usually detected in the first convolutional layer.
% In deeper convolutional layers, the network learns to detect more complicated features.
target_layer = 'activation_2_relu';

% Get the activations of the target layer
image_activations = activations(netTransfer,img_resized,target_layer);

% Display activations
figure
montage(image_activations);
    
%% Visualise extracted features using deep dream
% Specify target layer for activation visualisation.
target_layer = 'bn_conv1';

% Specify number of features to display.
channels = 1:25;

% Get extracted features.
features = deepDreamImage(netTransfer,target_layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0);

% Display extracted features.
figure
for i = 1:25
    subplot(5,5,i)
    imshow(features(:,:,:,i))
end

%% Read one image from the test dataset and classify
% Get an image
img = preprocess_image(testing_set.Files{5});
% Resize image to 224x224
img_resized = imresize(img,[224 224]);

% Classify image using trained CNN.
[class, score]=classify(netTransfer,img_resized);

% Display image and classification result
figure
imshow(img_resized)
title([ 'Predicted Class:' char(string(class)),', ','score=',num2str(max(score)),', ','Actual Class:', char(string(testing_set.Labels(1)))])