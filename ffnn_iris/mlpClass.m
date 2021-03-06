function [model, L] = mlpClass(X, y, k, lambda)
% Train a multilayer perceptron neural network for multiclass classification with backpropagation
% logistic activation function is used.
% Input:
%   X: d x n data matrix
%   y: 1 x n label vector
%   k: T x 1 vector to specify number of hidden nodes in each layer
%   lambda: regularization parameter
% Ouput:
%   model: model structure
%   L: (regularized cross entropy) loss
% Example:
%   X = [1     2;
%        4     2;
%        1     3;
%        4     3];
%   Y = [1; 1; 2; 2];
%   k = 2;
%   Assign the following weights and bias
%   W{1} = [-1.0292    0.0950;
%            0.5061    0.5725];
%   b{1} = [1.1012; -0.5105];
%   W{2} = [-1.1995   -0.2254;
%           -1.8149    1.3689];
%   b{2} = [0.0404; 1.2023];
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    lambda = 1e-2;
end
eta = 1e-3;%1e-3;
tol = 1e-4;%1e-4;
maxiter = 50000;
L = inf(1,maxiter);

Y = sparse(y,1:numel(y),1);
k = [size(X,1);k(:);size(Y,1)];
T = numel(k)-1;
W = cell(T,1);
b = cell(T,1);
for t = 1:T
    W_tmp = randn(k(t),k(t+1)); 
    W{t} = W_tmp;
    b_tmp = randn(k(t+1),1);
    b{t} = b_tmp;
end
%   W{1} = [-1.0292    0.0950;
%            0.5061    0.5725];
%   b{1} = [1.1012; -0.5105];
%   W{2} = [-1.1995   -0.2254;
%            -1.8149    1.3689];
%   b{2} = [0.0404; 1.2023];
R = cell(T,1);
Z = cell(T+1,1);
Z{1} = X;
for iter = 2:maxiter
%     forward
    for t = 1:T-1
        %Z{t+1} = sigmoid(W{t}'*Z{t}+b{t});         % 5.10 5.113
        Z{t+1} = tanh(W{t}'*Z{t}+b{t});         % 5.10 5.113
    end
    Z{T+1} = softmax(W{T}'*Z{T}+b{T});   
    
%     loss
    E = Z{T+1};
    Wn = cellfun(@(x) dot(x(:),x(:)),W);            % |W|^2
    L(iter) = -dot(Y(:),log(E(:)))+0.5*lambda*sum(Wn);
    if abs(L(iter)-L(iter-1)) < tol*L(iter-1); break; end

%     backward
    R{T} = Z{T+1}-Y;                
    for t = T-1:-1:1
        df = Z{t+1}.*(1-Z{t+1});    % h'(a)
        R{t} = df.*(W{t+1}*R{t+1});     % 5.66
    end
    
%     gradient descent
    for t=1:T
        dW = Z{t}*R{t}'+lambda*W{t};      % 5.67
        db = sum(R{t},2);
        W{t} = W{t}-eta*dW;               % 5.43
        b{t} = b{t}-eta*db;
    end
end
L = L(2:iter);
model.W = W;
model.b = b;
