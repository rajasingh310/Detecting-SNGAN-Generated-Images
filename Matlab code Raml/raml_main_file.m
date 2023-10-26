%{

~Main file

Description: To classify real and fake images
Author: Rajababu Singh
Matriculation number: 1530580
Supervisor: Dr. Margret Keuper
Affiliation: University of Siegen

%}

%%
clc
clear

%% create image datastore and label for training

% For real images
train_real_image_path = 'C:\Users\rajas\Downloads\raml_data\train\imagewoof\*.jpg';
train_real_image_imd = imageDatastore(train_real_image_path); % imd = imagedatastore
train_real_image_label = ones(size(train_real_image_imd.Files));

% For train on Bilinear upsampling techniques fake images
train_bilinear_fake_image_path = 'C:\Users\rajas\Downloads\raml_data\train\SNGAN_bilinear\*.jpg';
train_bilinear_fake_image_imd = imageDatastore(train_bilinear_fake_image_path);
train_bilinear_fake_image = readall(combine(train_real_image_imd, train_bilinear_fake_image_imd));
train_bilinear_fake_image = cat(4, train_bilinear_fake_image{:});

train_bilinear_fake_image_label = repmat(2, size(train_bilinear_fake_image_imd.Files));
train_bilinear_fake_image_label = categorical([train_real_image_label; train_bilinear_fake_image_label]);

% For train on Bicubic upsampling techniques fake images
train_bicubic_fake_image_path = 'C:\Users\rajas\Downloads\raml_data\train\SNGAN_bicubic\*.jpg';
train_bicubic_fake_image_imd = imageDatastore(train_bicubic_fake_image_path);
train_bicubic_fake_image = readall(combine(train_real_image_imd, train_bicubic_fake_image_imd));
train_bicubic_fake_image = cat(4, train_bicubic_fake_image{:});

train_bicubic_fake_image_label = repmat(2, size(train_bicubic_fake_image_imd.Files));
train_bicubic_fake_image_label = categorical([train_real_image_label; train_bicubic_fake_image_label]);

% For train on Pixelshuffle upsampling techniques fake images
train_pixelshuffle_fake_image_path = 'C:\Users\rajas\Downloads\raml_data\train\SNGAN_pixelshuffle\*.jpg';
train_pixelshuffle_fake_image_imd = imageDatastore(train_pixelshuffle_fake_image_path);
train_pixelshuffle_fake_image = readall(combine(train_real_image_imd, train_pixelshuffle_fake_image_imd));
train_pixelshuffle_fake_image = cat(4, train_pixelshuffle_fake_image{:});

train_pixelshuffle_fake_image_label = repmat(2, size(train_pixelshuffle_fake_image_imd.Files));
train_pixelshuffle_fake_image_label = categorical([train_real_image_label; train_pixelshuffle_fake_image_label]);

%%  create image datastore and label for validation

% For real images
val_real_image_path = 'C:\Users\rajas\Downloads\raml_data\test\imagewoof_test\*.jpg';
val_real_image_imd = imageDatastore(val_real_image_path);
val_real_image_label = ones(size(val_real_image_imd.Files));

% For validation on Bilinear upsampling techniques fake images
val_bilinear_fake_image_path = 'C:\Users\rajas\Downloads\raml_data\test\SNGAN_bilinear_test\*.jpg';
val_bilinear_fake_image_imd = imageDatastore(val_bilinear_fake_image_path);
val_bilinear_fake_image = readall(combine(val_real_image_imd, val_bilinear_fake_image_imd));
val_bilinear_fake_image = cat(4, val_bilinear_fake_image{:});

val_bilinear_fake_image_label = repmat(2, size(val_bilinear_fake_image_imd.Files));
val_bilinear_fake_image_label = categorical([val_real_image_label; val_bilinear_fake_image_label]);

% For validation on Bicubic upsampling techniques fake images
val_bicubic_fake_image_path = 'C:\Users\rajas\Downloads\raml_data\test\SNGAN_bicubic_test\*.jpg';
val_bicubic_fake_image_imd = imageDatastore(val_bicubic_fake_image_path);
val_bicubic_fake_image = readall(combine(val_real_image_imd, val_bicubic_fake_image_imd));
val_bicubic_fake_image = cat(4, val_bicubic_fake_image{:});

val_bicubic_fake_image_label = repmat(2, size(val_bicubic_fake_image_imd.Files));
val_bicubic_fake_image_label = categorical([val_real_image_label; val_bicubic_fake_image_label]);

% For validation on Pixelshuffle upsampling techniques fake images
val_pixelshuffle_fake_image_path = 'C:\Users\rajas\Downloads\raml_data\test\SNGAN_pixelshuffle_test\*.jpg';
val_pixelshuffle_fake_image_imd = imageDatastore(val_pixelshuffle_fake_image_path);
val_pixelshuffle_fake_image = readall(combine(val_real_image_imd, val_pixelshuffle_fake_image_imd));
val_pixelshuffle_fake_image = cat(4, val_pixelshuffle_fake_image{:});

val_pixelshuffle_fake_image_label = repmat(2, size(val_pixelshuffle_fake_image_imd.Files));
val_pixelshuffle_fake_image_label = categorical([val_real_image_label; val_pixelshuffle_fake_image_label]);

%% Call Network function

%{
[network_bilinear_rgb, network_bicubic_rgb, network_pixelshuffle_rgb, network_bicubic_rgb_transfer_learned, network_pixelshuffle_rgb_transfer_learned] = net( ...
    train_bilinear_fake_image, train_bicubic_fake_image, train_pixelshuffle_fake_image, train_bilinear_fake_image_label, train_bicubic_fake_image_label, train_pixelshuffle_fake_image_label, ...
    val_bilinear_fake_image, val_bicubic_fake_image, val_pixelshuffle_fake_image, val_bilinear_fake_image_label, val_bicubic_fake_image_label, val_pixelshuffle_fake_image_label);
%}

%{
%% call CNN model for rgb images

network_rgb_bilinear = cnn_raml_2(train_bilinear_fake_image, train_bilinear_fake_image_label, val_bilinear_fake_image, val_bilinear_fake_image_label);
network_rgb_bicubic = cnn_raml_2(train_bicubic_fake_image, train_bicubic_fake_image_label, val_bicubic_fake_image, val_bicubic_fake_image_label);
network_rgb_pixelshuffle = cnn_raml_2(train_pixelshuffle_fake_image, train_pixelshuffle_fake_image_label, val_pixelshuffle_fake_image, val_pixelshuffle_fake_image_label);

%% Transfer Learning

network_rgb_bicubic_transfer_learned = trainNetwork(train_bicubic_fake_image, train_bicubic_fake_image_label, network_rgb_bilinear.Layers, options_hyperparameter(val_bicubic_fake_image, val_bicubic_fake_image_label));
%%
network_rgb_pixelshuffle_transfer_learned = trainNetwork(train_pixelshuffle_fake_image, train_pixelshuffle_fake_image_label, network_rgb_bilinear.Layers, options_hyperparameter(val_pixelshuffle_fake_image, val_pixelshuffle_fake_image_label));

%}


%% transform the training and validation images into grayscale

% Bilinear
train_bilinear_fake_image_gray = gray_scale(train_bilinear_fake_image);
val_bilinear_fake_image_gray = gray_scale(val_bilinear_fake_image);

% Bicubic
train_bicubic_fake_image_gray = gray_scale(train_bicubic_fake_image);
val_bicubic_fake_image_gray = gray_scale(val_bicubic_fake_image);

% Pixelshuffle
train_pixelshuffle_fake_image_gray = gray_scale(train_pixelshuffle_fake_image);
val_pixelshuffle_fake_image_gray = gray_scale(val_pixelshuffle_fake_image);

%% Call network function

%{
[network_bilinear_gray, network_bicubic_gray, network_pixelshuffle_gray, network_bicubic_gray_transfer_learned, network_pixelshuffle_gray_transfer_learned] = net( ...
    train_bilinear_fake_image_gray, train_bicubic_fake_image_gray, train_pixelshuffle_fake_image_gray, train_bilinear_fake_image_label, train_bicubic_fake_image_label, train_pixelshuffle_fake_image_label, ...
    val_bilinear_fake_image_gray, val_bicubic_fake_image_gray, val_pixelshuffle_fake_image_gray, val_bilinear_fake_image_label, val_bicubic_fake_image_label, val_pixelshuffle_fake_image_label);
%}

%{
%%

train1_gray = transform(train_real_image_imd,@(x) im2gray(x));
train2_gray = transform(train_bilinear_fake_image_imd,@(x) im2gray(x));

i = 1;
img1 = cell(hasdata(train1_gray) , 1);
img2 = cell(hasdata(train1_gray) , 1);
while hasdata(train1_gray) 
    img1{i} = read(train1_gray) ;    % read image from datastore
    img2{i} = read(train2_gray) ; 
    i = i + 1;
end
img1 = img1(:);
img2 = img2(:);

x_train_gray = cat(4, img1{:}, img2{:});

%% transform the test images into grayscale

test1_gray = transform(val_real_image_imd,@(x) im2gray(x));
test2_gray = transform(val_bilinear_fake_image_imd,@(x) im2gray(x));

i = 1;
img3 = cell(hasdata(test1_gray), 1);
img4 = cell(hasdata(test1_gray), 1);

while hasdata(test1_gray) 
    img3{i} = read(test1_gray) ;    % read image from datastore
    img4{i} = read(test2_gray) ; 
    i = i + 1;
end
img3 = img3(:);
img4 = img4(:);

x_test_gray = cat(4, img3{:}, img4{:});

%% call CNN model

net_gray = algorithm_FCNN_small(x_train_gray, x_test_gray, y_train_bilinear, y_val_bilinear_fake_image);


%}

%% transform the training and validation images into FFT

% Bilinear
train_bilinear_fake_image_FFT = FF_Transform(train_bilinear_fake_image_gray);
val_bilinear_fake_image_FFT = FF_Transform(val_bilinear_fake_image_gray);

% Bicubic
train_bicubic_fake_image_FFT = FF_Transform(train_bicubic_fake_image_gray);
val_bicubic_fake_image_FFT = FF_Transform(val_bicubic_fake_image_gray);

% Pixelshuffle
train_pixelshuffle_fake_image_FFT = FF_Transform(train_pixelshuffle_fake_image_gray);
val_pixelshuffle_fake_image_FFT = FF_Transform(val_pixelshuffle_fake_image_gray);

%% Call network function

[network_bilinear_FFT, network_bicubic_FFT, network_pixelshuffle_FFT, network_bicubic_transfer_learned_FFT, network_pixelshuffle_transfer_learned_FFT] = net( ...
    train_bilinear_fake_image_FFT, train_bicubic_fake_image_FFT, train_pixelshuffle_fake_image_FFT, train_bilinear_fake_image_label, train_bicubic_fake_image_label, train_pixelshuffle_fake_image_label, ...
    val_bilinear_fake_image_FFT, val_bicubic_fake_image_FFT, val_pixelshuffle_fake_image_FFT, val_bilinear_fake_image_label, val_bicubic_fake_image_label, val_pixelshuffle_fake_image_label);


%{
%% call function ff_Transform to transform image data

% transform training data
transformed_img1_data = ff_Transform(img1);
transformed_img2_data = ff_Transform(img2);

% transform test data
transformed_img3_data = ff_Transform(img3);
transformed_img4_data = ff_Transform(img4);

x_train_fft2 = cat(4, transformed_img1_data{:}, transformed_img2_data{:});
x_test_fft2 = cat(4, transformed_img3_data{:}, transformed_img4_data{:});

%% call FCNN model

net_fft2 = algorithm_FCNN_small(x_train_fft2, x_test_fft2, y_train_bilinear, y_val_bilinear_fake_image);

% Other useful networks
%net_fft2 = algorithm_cnn1(x_train_fft2, x_test_fft2, y_train, y_val);
%net_fft2 = algorithm_cnn2(x_train_fft2, x_test_fft2, y_train, y_val);
%net_fft2 = algorithm_cnn4(x_train_fft2, x_test_fft2, y_train, y_val);
%net_fft2 = algorithm_fcnn(x_train_fft2, x_test_fft2, y_train, y_val);
%net_fft2 = algorithm_ResNet(x_train_fft2, x_test_fft2, y_train, y_val);

%% calculate accuracy

Ypreds = double(classify(net_fft2, x_test_fft2));
Ytest = double(y_val_bilinear_fake_image);
accuracy = int32(100*sum(Ypreds == Ytest)/numel(Ytest));

%% plot figure confusion chart

confusionchart(Ytest, Ypreds)

%}

%% Function to call Network

function [network_bilinear, network_bicubic, network_pixelshuffle, network_bicubic_transfer_learned, network_pixelshuffle_transfer_learned] = net( ...
    train_bilinear, train_bicubic, train_pixelshuffle, train_bilinear_label, train_bicubic_label, train_pixelshuffle_label, ...
    val_bilinear, val_bicubic, val_pixelshuffle, val_bilinear_label, val_bicubic_label, val_pixelshuffle_label)

    % call CNN model for rgb images

    network_bilinear = cnn_raml_2(train_bilinear, train_bilinear_label, val_bilinear, val_bilinear_label);
    network_bicubic = cnn_raml_2(train_bicubic, train_bicubic_label, val_bicubic, val_bicubic_label);
    network_pixelshuffle = cnn_raml_2(train_pixelshuffle, train_pixelshuffle_label, val_pixelshuffle, val_pixelshuffle_label);
    
    %% Transfer Learning
    
    % Bicubic
    network_bicubic_transfer_learned = trainNetwork(train_bicubic, train_bicubic_label, network_bilinear.Layers, options_hyperparameter(val_bicubic, val_bicubic_label));
    
    % Pixelshuffle
    network_pixelshuffle_transfer_learned = trainNetwork(train_pixelshuffle, train_pixelshuffle_label, network_bilinear.Layers, options_hyperparameter(val_pixelshuffle, val_pixelshuffle_label));

end


%% function options for transfer learning

function fitted_options = options_hyperparameter(val_images, val_labels)
    
    fitted_options = trainingOptions('adam',...    
        'InitialLearnRate',0.2,... 
        'LearnRateSchedule', 'piecewise',... 
        'LearnRateDropFactor', 0.65,...
        'LearnRateDropPeriod', 1,...    
        'MaxEpochs',500,...           
        'MiniBatchSize', 75,...  
        'Shuffle', 'every-epoch',...
        'ValidationData', {val_images, val_labels},...
        'ValidationFrequency', 200, ...
        'ValidationPatience', 5, ...
        'VerboseFrequency', 200, ...
        'plots','training-progress',...
        'ExecutionEnvironment','gpu');

end

%% Function convert images to gray-scale

function gray_scale_images = gray_scale(images)

    gray_scale_images = zeros(size(images,1), size(images,2), 1, size(images,4));
    for i = 1: size(images,4)
        gray_scale_images(:,:,1,i) = rgb2gray(images(:,:,:,i));
    end

end

%% function ff_Transform

function FFT_transformed_image_data = FF_Transform(image_data)
    
    FFT_transformed_image_data = zeros(size(image_data,1), size(image_data,2), 4, size(image_data,4));
    for i = 1: size(image_data, 4)    % loop over all images
      
        % create channels consists of real value, imaginary value,
        % absolute, phase
        FFT_transformed_image_data(:,:,1,i)= abs(fft2(image_data(:,:,:,i)));
        FFT_transformed_image_data(:,:,2,i)= angle(fft2(image_data(:,:,:,i)));
        FFT_transformed_image_data(:,:,3,i)= real(fft2(image_data(:,:,:,i)));
        FFT_transformed_image_data(:,:,4,i)= imag(fft2(image_data(:,:,:,i)));
    
    end

end