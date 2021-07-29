function [y ypred] = rbfClassify(X, model);
% This code is for educational and research purposes of comparisons. This
% is a RBF Neural Network with four layers on a three class 
% iris data set.
%
% This code is modified from the old version of Matlab's RBF
%
% [y ypred] = rbfMatlabClassifyBRodriguez(X, y, spread)
%
% This function returns the y labels as an approximation under the gaussian
% curve while the returned value ypred returns class labels as [-1 1]
%
% Input
%   X [n x d] data to be classified with n observations and dimension d
%   model - structure containing:
%     .W_hat - layer weights
%     .W - input weights
%     .bias
%     .spread
%     .input_spread
%     .error - training error [0 1]
%
% Output
%   y [n x 1]labels as an approximation under the gaussian curve
%   ypred [n x 1] class labels [-1 1]

[n1, d1] = size(X); X = X';
[n2, d2] = size(model.W');

H = zeros(n1, n2);
for j = 1:n2
    W = model.W(:,j);
    D = X - W(:,ones(1,n1));
    D = D.*model.spread;
    s = multiDiag(D',D);
	H(:, j) = exp(-s);
end

y = (H * model.W_hat')' + model.bias;
ypred = ones(size(y));
ypred(find(y<0)) = -1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function returns the diagonal product of X1 and X2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xDiag = multiDiag(X1, X2)
% Inputs
%   X1 - [d x n]
%   X2 - [n x d]
%
% Output
%   xDiag - [d x 1] 

[r1,c1] = size(X1);
[r2,c2] = size(X2);

X1tmp = X1';
X1tmp = X1tmp(:);
X2tmp = X2(:);
X = zeros(c1,r1);
X(:) = X1tmp .* X2tmp;
[r1,c1] = size(X);
if r1 > 1
	xDiag = sum(X)';
else
	xDiag = X';
end
