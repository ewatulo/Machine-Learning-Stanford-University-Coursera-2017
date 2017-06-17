function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
val_errors = zeros(size(c_values, 2), size(c_values, 2));

for c=1:size(c_values, 2)
    for s=1:size(sigma_values, 2)
        model= svmTrain(X, y, c_values(c), @(x1, x2) gaussianKernel(x1, x2, sigma_values(s)));
        predictions = svmPredict(model, Xval);
        val_errors(c, s) = mean(double(predictions ~= yval));
    end
end    
[r,c] = find(val_errors==min(val_errors(:)))
C = c_values(r);
sigma = sigma_values(c);
% =========================================================================

end
