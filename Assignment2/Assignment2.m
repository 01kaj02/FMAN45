%% Assignment 2
% Kajsa Hansson Willis
% FMAN45 Machine Learning 

clear;
addpath("Code stub (for students)/Matlab/");
load('A2_data.mat')

%% Exercise 7

X = train_data_01-mean(train_data_01,2); % zero mean required for PCA
[U, ~, ~] = svd(X);

PC = U(:, 1:2)';
proj =  PC * X;

figure;
gscatter(proj(1, :)', proj(2, :)', train_labels_01, 'br','ox');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Linear PCA of MNIST');
legend('0', '1');
grid on;


%% Exercise 8 

K = 2;
k = 5;

[y2,C2] = K_means_clustering(train_data_01,K);

X2 = train_data_01-mean(train_data_01); % zero mean required for PCA
[U2, ~, ~] = svd(X2);

PC2 = U2(:, 1:2)';
proj2 =  PC2 * X2;

figure;
gscatter(proj2(1, :)', proj2(2, :)', y2, 'br','ox');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Linear PCA of Exercise 8 with two clusters');
legend('0', '1');
grid on;


[y5, C5] = K_means_clustering(train_data_01, k);

X5 = train_data_01-mean(train_data_01); 
[U5, S5, V5] = svd(X5);

PC5 = U5(:, 1:2)';
proj5 =  PC5 * X5;

figure;
gscatter(proj5(1, :)', proj5(2, :)', y5, 'brgmk','oxv*.');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Linear PCA of Exercise 8 with five clusters');
legend('0', '1');
grid on;

%% Exercise 9 

imgC2a = reshape(C2(:, 1), [28 28]);
imgC2b = reshape(C2(:, 2), [28 28]);

figure
hold on
subplot(1,2,1);
imshow(imgC2a);
title('Cluster 1')
subplot(1,2,2);
imshow(imgC2b);
title('Cluster 2')
hold off

imgC5a = reshape(C5(:, 1), [28 28]);
imgC5b = reshape(C5(:, 2), [28 28]);
imgC5c = reshape(C5(:, 3), [28 28]);
imgC5d = reshape(C5(:, 4), [28 28]);
imgC5e = reshape(C5(:, 5), [28 28]);

figure
hold on
subplot(1,5,1);
imshow(imgC5a);
title('Cluster 1')
subplot(1,5,2);
imshow(imgC5b);
title('Cluster 2')
hold off
subplot(1,5,3);
imshow(imgC5c);
title('Cluster 3')
hold off
subplot(1,5,4);
imshow(imgC5d);
title('Cluster 4')
hold off
subplot(1,5,5);
imshow(imgC5e);
title('Cluster 5')
hold off


%% Exercise 10

lgthtrain = length(train_data_01);
lgthtest = length(test_data_01);

clustertrain = zeros(lgthtrain,1);
clustertest = zeros(lgthtest,1);

for i=1:lgthtrain
 clustertrain(i) = K_means_classifier(train_data_01(:,i), C2);
end 

for i=1:lgthtest 
 clustertest(i) = K_means_classifier(test_data_01(:,i), C2);
end 

clusterlabelstrain = zeros(2, 1);  

for k = 1:2
    index = find(clustertrain == k);              
    truelabels = train_labels_01(index);   
    clusterlabelstrain(k) = mode(truelabels);  
    fprintf('Cluster %d assigned to class: %d\n', k, clusterlabelstrain(k)); % samma gäller för test data också  
end

predTrainLabels = clusterlabelstrain(clustertrain);
predTestLabels = clusterlabelstrain(clustertest);

[trainRes, wrongtrain] = evaluator(predTrainLabels, train_labels_01);
trainRate = 100 * wrongtrain / lgthtrain;

[testRes, wrongtest] = evaluator(predTestLabels, test_labels_01);
testRate = 100 * wrongtest / lgthtest;

disp(trainRes) % obs det blir flippat pga ettor och nollor flippas
fprintf('Misclassified with training data: %d out of %d (%.2f%%)\n\n', ...
    wrongtrain, lgthtrain, trainRate);

disp(testRes)
fprintf('Misclassified with testing data: %d out of %d (%.2f%%)\n', ...
    wrongtest, lgthtest, testRate);


%% Plot Exercise 10 (Training data)

datamean = mean(train_data_01, 2);
Xcenter = train_data_01 - datamean;

[U, ~, ~] = svd(Xcenter);

projX = U(:, 1:2)' * Xcenter;  

C2center = C2 - datamean;  
projC2 = U(:, 1:2)' * C2center;  

figure;
gscatter(projX(1, :)', projX(2, :)', clustertrain, 'rb', 'ox');
hold on;
plot(projC2(1, :), projC2(2, :), 'kp', 'MarkerSize', 15, 'LineWidth', 2);  
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('K-means Clustering for Classification PCA');
grid on;

wrongvalues = find(predTrainLabels ~= train_labels_01);
plot(projX(1, wrongvalues), projX(2, wrongvalues), 'k.', 'MarkerSize', 10);
legend('Cluster 1', 'Cluster 2', 'the centroids', 'Incorrect predictions');

 %% Exercise 11 K=5

 K=5;

lgthtrain = length(train_data_01);
lgthtest = length(test_data_01);

clustertrain = zeros(lgthtrain,1);
clustertest = zeros(lgthtest,1);

for i=1:lgthtrain
 clustertrain(i) = K_means_classifier(train_data_01(:,i), C5); % uppdatera
end 

for i=1:lgthtest 
 clustertest(i) = K_means_classifier(test_data_01(:,i), C5); % uppdatera
end 

clusterlabelstrain = zeros(K, 1);  

for k = 1:K
    indices = find(clustertrain == k);              
    true_labels_k = train_labels_01(indices);   
    clusterlabelstrain(k) = mode(true_labels_k);  
    fprintf('Cluster %d assigned to class: %d\n', k, clusterlabelstrain(k));

end 

predTrainLabels = clusterlabelstrain(clustertrain);
predTestLabels = clusterlabelstrain(clustertest);

[trainRes, wrongtrain] = evaluator(predTrainLabels, train_labels_01);
trainRate = 100 * wrongtrain / lgthtrain;

[testRes, wrongtest] = evaluator(predTestLabels, test_labels_01);
testRate = 100 * wrongtest / lgthtest;

disp(trainRes) % obs det blir flippat pga ettor och nollor flippas
fprintf('Misclassified with training data: %d out of %d (%.2f%%)\n\n', ...
    wrongtrain, lgthtrain, trainRate);

disp(testRes)
fprintf('Misclassified with testing data: %d out of %d (%.2f%%)\n', ...
    wrongtest, lgthtest, testRate);

%% Exercise 11 K=3, 10, 20, 30

K = 30; 

[~, Ck] = K_means_clustering(train_data_01, K);

clustertrain = zeros(lgthtrain,1);
clustertest = zeros(lgthtest,1);

for i=1:lgthtrain
 clustertrain(i) = K_means_classifier(train_data_01(:,i), Ck); % uppdatera
end 

for i=1:lgthtest 
 clustertest(i) = K_means_classifier(test_data_01(:,i), Ck); % uppdatera
end 

clusterlabelstrain = zeros(K, 1);  

for k = 1:K
    indices = find(clustertrain == k);              
    true_labels_k = train_labels_01(indices);   
    clusterlabelstrain(k) = mode(true_labels_k);  
    fprintf('Cluster %d assigned to class: %d\n', k, clusterlabelstrain(k));

end 

predTrainLabels = clusterlabelstrain(clustertrain);
predTestLabels = clusterlabelstrain(clustertest);

[trainRes, wrongtrain] = evaluator(predTrainLabels, train_labels_01);
trainRate = 100 * wrongtrain / lgthtrain;

[testRes, wrongtest] = evaluator(predTestLabels, test_labels_01);
testRate = 100 * wrongtest / lgthtest;

disp(trainRes) % obs det blir flippat pga ettor och nollor flippas
fprintf('Misclassified with training data: %d out of %d (%.2f%%)\n\n', ...
    wrongtrain, lgthtrain, trainRate);

disp(testRes)
fprintf('Misclassified with testing data: %d out of %d (%.2f%%)\n', ...
    wrongtest, lgthtest, testRate);


%% Exercise 12 

model12 = fitcsvm(train_data_01',train_labels_01);

predicttrain = predict(model12,train_data_01');
predicttest = predict(model12,test_data_01');

[train_errors, wrongtrain] = evaluator(predicttrain, train_labels_01);
[test_errors, wrongtest] = evaluator(predicttest, test_labels_01);


%% Exercise 13 


beta = 100; % change to lower missclassifiction

%model13 = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian');
model13 = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian','KernelScale', beta);

predicttest13 = predict(model13,test_data_01');
[test_errors13, wrongtest13] = evaluator(predicttest13, test_labels_01)

predicttrain13 = predict(model13,train_data_01');
[train_errors13, wrongtrain13] = evaluator(predicttrain13, train_labels_01);

