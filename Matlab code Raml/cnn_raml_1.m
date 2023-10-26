function net = cnn_raml_1(x_train, y_train, x_val, y_val)

    layers = [
         imageInputLayer(size(x_train, 1, 2, 3),"Name","imageinput")

         convolution2dLayer([1 3],4,"Name","conv_1","Padding","same")
         reluLayer("Name","relu_1")
         dropoutLayer(0.2)
    
         fullyConnectedLayer(16,"Name","fc_1")
         reluLayer("Name","relu_4")
         fullyConnectedLayer(numel(unique(y_train)),"Name","fc_3")
         softmaxLayer("Name","softmax")
         classificationLayer("Name","classoutput")
         ];
    
    options = trainingOptions('adam',...    
        'InitialLearnRate',0.08,... 
        'LearnRateSchedule', 'piecewise',... 
        'LearnRateDropFactor', 0.65,...
        'LearnRateDropPeriod', 1,...    
        'MaxEpochs',500,...           
        'MiniBatchSize', 75,...  
        'Shuffle', 'every-epoch',...
        'ValidationData', {x_val, y_val},...
        'ValidationFrequency', 200, ...
        'ValidationPatience', 5, ...
        'VerboseFrequency', 200, ...
        'plots','training-progress',...
        'ExecutionEnvironment','gpu');
    
    net = trainNetwork(x_train, y_train, layers, options);

end