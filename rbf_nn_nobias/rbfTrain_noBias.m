function model = rbfTrainBRodriguez(X, y, spread)
% This code is for educational and research purposes of comparisons. This
% is a RBF Neural Network with four layers on a three class 
% iris data set. In this implementation there is  no bias term. 
%
% This code is modified from the old version of Matlab's RBF
%
% model = rbfTrainBRodriguez(X, y, spread)
%
% Gets the design matrix from the input data, centre positions
% and radii factors.
%
% Input
%       X [n x d] training data with n observations and dimension d
%       y [n x 1] labeled targets for classification two class [-1 1]
%       spread	
%
% Output
%   model - structure containing:
%     .W_Hat - layer weights
%     .W - input weights
%     .spread
%     .error - training error [0 1]
if nargin < 3
    spread = 0.5;
end

[n, d] = size(X); X = X';

H = zeros(n, n);
for j = 1:n
    W = X(:,j);
    D = X - W(:,ones(1,n));
	s = multiDiag(D',D)/(2*spread^2);
	H(:, j) = exp(-s);
end

%https://www.mathworks.com/help/matlab/ref/pinv.html
W_hat = pinv(H' * H) * H' * y; 
yt = (H * W_hat)';
ypred = ones(size(y));
ypred(find(yt<0)) = -1;
predError = 1 - length(y == ypred)/size(y,1);

model.W_hat = W_hat;
model.W = X;
model.spread = spread;
model.error = predError;
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
