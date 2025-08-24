%% Assignment 3 
% Kajsa Hansson Willis
% FMAN45 Machine Learning 

clear;
addpath("data");
addpath("layers");
addpath("models");
addpath("tests");

%% Exercise 2

test_fully_connected

%% Exercise 3

test_relu

%% Exercise 4

test_softmaxloss
test_gradient_whole_net

%% Exercise 5

mnist_starter()


%% Exercise 6
load models/network_trained_with_momentum.mat % the saved model from the mnist_starter

% Plotting the convolutional layer learns
convlayer = net.layers{1, 2};
convweights = convlayer.params.weights;

batchsize = 16; % the number of parameters in the convolution layer 
for l = 1:batchsize
    subplot(4, 4, l)
    imshow(convweights(:,:,l))
end

% Finding the misclassified and plotting them

x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
y_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');

y_test(y_test==0) = 10;

x_test = reshape(x_test, [28, 28, 1, 10000]);

pred = zeros(numel(y_test),1);
    for i=1:batchsize:size(y_test)
        idx = i:min(i+batchsize-1, numel(y_test));
        y = evaluate(net, x_test(:,:,:,idx), y_test(idx));
        [~, p] = max(y{end-1}, [], 1);
        pred(idx) = p;
    end
 fprintf('Accuracy on the test set: %f\n', mean(vec(pred) == vec(y_test)));

pred(pred == 10) = 0;
y_test(y_test == 10) = 0;
misclassidx = find(pred ~= y_test, 16);

figure;
for l = 1:16
    subplot(4, 4, l)
    imshow(x_test(:, :, :, misclassidx(l)));
    tit = strcat([...
        'True label: ',...
        num2str(y_test(misclassidx(l))),...
        newline,...
        'Predicted label: ',...
        num2str(pred(misclassidx(l)))...
        ]);
    title(tit);
end

% Plot the confusion matrix 
figure; 
confusion = confusionmat(y_test, pred);
confusionchart(confusion);

% The precision and the recall for all the digits 
precision = zeros(1,10);
recall = zeros(1,10);
for l=1:10
    correct = confusion(l, l);
    n = sum(confusion(:, l));
    precision(l) = correct / n;

    predictions = sum(confusion(l, :));
    recall(l) = correct / predictions; 
end 


% Determine number of parameters
nolayers = 9; 

weights = zeros(nolayers,1);
biases = zeros(nolayers,1);

for layeridx=1:nolayers
    layer = net.layers{1, layeridx};
    weights(layeridx) = 0;
    biases(layeridx) = 0;
    if isfield(layer, 'params')
        if isfield(layer.params, 'weights')
            weights(layeridx) = numel(layer.params.weights);
        end
        if isfield(layer.params, 'biases')
            biases(layeridx) = numel(layer.params.biases);
        end
    end
end

weights_total = sum(weights);
biases_total = sum(biases);


 %% Exercise 7 intro

 cifar10_starter()

 %% Exercise 7 param for 

 % Determine number of parameters
load('/Users/khw/Documents/År 4/Machine learning/Assignments/Assignment 3/Matlab/models/cifar10_baseline.mat') % the saved model from the improved_cifar10_starter

nolayers = 8; 

weights = zeros(nolayers,1);
biases = zeros(nolayers,1);

for layeridx=1:nolayers
    layer = net.layers{1, layeridx};
    weights(layeridx) = 0;
    biases(layeridx) = 0;
    if isfield(layer, 'params')
        if isfield(layer.params, 'weights')
            weights(layeridx) = numel(layer.params.weights);
        end
        if isfield(layer.params, 'biases')
            biases(layeridx) = numel(layer.params.biases);
        end
    end
end

weights_total = sum(weights);
biases_total = sum(biases);


  %% Exercise 7 cont. 
 improved_cifar10_starter()


 %% Exercise 7 cont. 

load('/Users/khw/Documents/År 4/Machine learning/Assignments/Assignment 3/Matlab/models/cifar10_baselinemod.mat') % the saved model from the improved_cifar10_starter

[x_train, y_train, x_test, y_test, classes] = load_cifar10(4);
data_mean = mean(mean(mean(x_train, 1), 2), 4); % mean RGB triplet
x_testnorm = bsxfun(@minus, x_test, data_mean);


% Plotting the convolutional layer learns
convlayer = net.layers{1, 2};
convweights = convlayer.params.weights;

batchsize = 16;
figure;
for l = 1:batchsize
    w = convweights(:,:,:,l); 
    w = (w - min(w(:))) / (max(w(:)) - min(w(:)) + eps); %normalize for imshow 
    subplot(4, 4, l);
    imshow(w);
    title(['RGB filter ' num2str(l)]);
end


% Finding the misclassified and plotting them

pred = zeros(numel(y_test),1);
    for i=1:batchsize:size(y_test)
        idx = i:min(i+batchsize-1, numel(y_test));
        y = evaluate(net, x_testnorm(:,:,:,idx), y_test(idx));
        [~, p] = max(y{end-1}, [], 1);
        pred(idx) = p;
    end

misclassidx = find(pred ~= y_test, 16);

figure;
for l = 1:16
    w = x_test(:, :, :, misclassidx(l)); 
    w = (w - min(w(:))) / (max(w(:)) - min(w(:)) + eps); %normalize for imshow 
    subplot(4, 4, l)
    imshow(w);
    tit = strcat([...
        'True label: ',...
        classes(y_test(misclassidx(l))),...
        newline,...
        'Predicted label: ',...
        classes(pred(misclassidx(l)))...
        ]);

    title(tit, 'FontSize', 9);
end

% Plot the confusion matrix 
figure; 
confusion = confusionmat(double(y_test), pred);
confusionchart(confusion, classes);

% The precision and the recall for all the digits 
precision = zeros(1,10);
recall = zeros(1,10);
for l=1:10
    correct = confusion(l, l);
    n = sum(confusion(:, l));
    precision(l) = correct / n;

    predictions = sum(confusion(l, :));
    recall(l) = correct / predictions; 
end 


% Determine number of parameters

nolayers = 9; % including the added convolution layer

weights = zeros(nolayers,1);
biases = zeros(nolayers,1);

for layeridx=1:nolayers
    layer = net.layers{1, layeridx};
    weights(layeridx) = 0;
    biases(layeridx) = 0;
    if isfield(layer, 'params')
        if isfield(layer.params, 'weights')
            weights(layeridx) = numel(layer.params.weights);
        end
        if isfield(layer.params, 'biases')
            biases(layeridx) = numel(layer.params.biases);
        end
    end
end

weights_total = sum(weights);
biases_total = sum(biases);
