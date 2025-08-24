%% Assignment 1
% Kajsa Hansson Willis
% FMAN45 Machine Learning

clear; 
addpath('Code stub (for students)/Matlab')
load 'A1_data'


%% Exercise 4

N = 50;

lambda = [0.1 10 0.75];

for i=1:3
    what = lasso_ccd(t, X, lambda(i)); 
    y_data = X * what;
    y_hat = Xinterp * what;

    figure
    hold on
    scatter(n, t, 25, 'b') % original 
    scatter(n, y_data, 25, 'filled') % reconstructed data points
    plot(ninterp, y_hat, 'r') % interpolated reconstruction 
    legend('Real Data', 'Reconstructed Data Points', 'Reconstruction')
    xlabel('Time')
    title('Reconstruction plot for \lambda = ',lambda(i))
    nonZero = sum(what~=0); 
    fprintf('Amount of nonzero: %d for lambda: %.3f\n', nonZero, lambda(i));
end 

lambdauser = 1; 


%% Exercise 5 

lambdaMin = 0.1;
lambdaMax = max(abs(X'*t));
Nlambda = 20;
lambdaGrid = exp(linspace(log(lambdaMin), log(lambdaMax), Nlambda));

K = 10; 

[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambdaGrid,K);

figure 
title('RMSE of lambdas')
xlabel('Lambda') 
ylabel('RMSE')
hold on
plot(lambdaGrid, RMSEval, '-x');
plot(lambdaGrid, RMSEest, '-o');
plot([lambdaopt lambdaopt], [0, max(RMSEval)], '--'); 
xlim([lambdaMin,lambdaMax])
legend({'RMSE for validation','RMSE for estimation', 'Optimal lambda'})
hold off

yrecon = X * wopt;
yhatrecon = Xinterp * wopt;

figure 
title('Reconstruction plot of optimal lambda')
hold on
scatter(n, t, 30, 'b') % original 
scatter(n, yrecon, 30, 'filled') % reconstructed data points
plot(ninterp, yhatrecon, 'r') % interpolated reconstruction 
legend('Real Data', 'Reconstructed Data Points', 'Reconstruction')
xlabel('Time')



%% Exercise 6 

lambdamin = 0.0001;
lambdamax = 0;
for i=1:56
    if i < 56
        t = Ttrain(352*(i-1)+1:352*i);
    else 
        t = Ttrain(19053:end);
    end 
    lambda = max(abs(Xaudio'*t));
    if lambda > lambdamax
        lambdamax = lambda; 
    end 
end 

nlambda = 40;
lambdav = exp(linspace(log(lambdamin), log(lambdamax), nlambda));
K = 5; 

[Wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain,Xaudio,lambdav,K);

figure 
title('RMSE of lambdas')
xlabel('Lambda') 
ylabel('RMSE')
hold on
plot(lambdav, RMSEval, '-x');
plot(lambdav, RMSEest, '-o');
plot([lambdaopt lambdaopt], [0, max(RMSEval)], '--'); 
xlim([lambdamin,lambdamax])
legend({'RMSE for validation','RMSE for estimation', 'Optimal lambda'})
hold off



%% Exercise 7 

%lambda = 0.0004;

soundsc(Ttrain,fs);
Yclean = lasso_denoise(Ttest, Xaudio, lambdaopt);
%Yclean = lasso_denoise(Ttest, Xaudio, lambda);
soundsc(Yclean,fs);

%save('denoised audio','Yclean','fs');


