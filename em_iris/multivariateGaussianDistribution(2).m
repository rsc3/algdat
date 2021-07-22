function y = multivariateGaussianDistribution(X, Mean, Cov)
% This code has been modified from its original content for educational
% and research purposes of comparisons. Full credit is given to the  
% original referenced developer. 
%
% This function evaluates a multi-variate Gaussian 
%  probability density function(s) for given input column vectors in X.
%
% Input:
%  X [dim x numData]
%  Mean [dim x ncomp]
%  Cov [dim x dim x ncomp]
%
% Output:
%  y [ncomp x numData] probability values
%  
% Reference:
%  
%  V. Franc, Optimization Algorithms for Kernal Methods, (2005)
%  V. Franc, Statistical Pattern Recognition Toolbox, (2007)
%  https://cmp.felk.cvut.cz/cmp/software/stprtool/

[dim,numData] = size(X);
ncomp = size(Mean,2);

if size(Cov,1) ~= size(Cov,2), Cov = reshape(Cov,1,1,ncomp); end

y = zeros(ncomp,numData);

% pdf evaluation for each class label
for i=1:ncomp,
  dist = mahalanobisDist(X,Mean(:,i),Cov(:,:,i));
  y(i,:) = exp(-0.5*dist)/sqrt((2*pi)^dim*det(Cov(:,:,i)));
end