%% Visualise the activations of a layer
% Get and image from the testing set
img = preprocess_image(testing_set.Files{5});

% Resize image to 224x224
img_resized = imresize(img,[224 224]);

% Specify target layer for activation visualisation.
% Features like edges are usually detected in the first convolutional layer.
% In deeper convolutional layers, the network learns to detect more complicated features.

%target_layer = 'conv1'; % first convolutional layer
%target_layer = 'res2a_branch1'; % after first convolutional block
target_layer = 'res5a_branch1'; % after last convolutional block

% Get the activations of the target layer
image_activations = activations(netTransfer,img_resized,target_layer);

% Display activations
figure
montage(image_activations)