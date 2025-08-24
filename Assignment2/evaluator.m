function [results, wrongs] = evaluator(prediction, true)

    pred_0 = true(prediction == 0);
    pred_1 = true(prediction == 1);

    results = zeros(2,2);
    results(1,1) = sum(pred_0 == 0); % correctly predicted zeros
    results(1,2) = sum(pred_0 == 1); % incorrectly predicted zeros
    results(2,1) = sum(pred_1 == 0); % incorrectly predicted ones
    results(2,2) = sum(pred_1 == 1); % correctly predicted ones
    wrongs = results(2,1) + results(1,2);
end