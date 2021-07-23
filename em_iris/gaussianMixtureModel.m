function model = gaussianMixtureModel(X,covType)
% This code has been modified from its original content for educational
% and research purposes of comparisons. Full credit is given to the  
% original referenced developer. 
%
% This function trains a model that computs the Maximal Likelihood 
% estimation of Gaussian mixture model for a given data set.
% 
%  model = gaussianMixtureModel(X)
%  model = gaussianMixtureModel(X,covType)
%
% Input:
%   X [dim x numData] Data sample.
%   data.X [dim x numData] Data sample.
%   data.y [1 x numData] Data labels.
%   covType [string] - Type of covariacne matrix
%                      covType = 'full'      full cov matrix
%                      covType = 'diag'      diagonal cov matrix
%                      covType = 'spherical' spherical cov matrix
%
% Output:
%   model - is a struct with the estimated Gaussian mixture model
%           parameters
%        .Mean  [dim x ncomp]
%        .Cov   [dim x dim x ncomp]
%        .Prior [1 x ncomp]
%  
% Reference:
%  
%  V. Franc, Optimization Algorithms for Kernal Methods, (2005)
%  V. Franc, Statistical Pattern Recognition Toolbox, (2007)
%  https://cmp.felk.cvut.cz/cmp/software/stprtool/

if ~isstruct(X),
  data.X = X;
  data.y = ones(1,size(data.X,2));
end
 
if nargin < 2, covType = 'full'; end

[dim,numData] = size(data.X);

labels = unique(data.y);
model.Mean = zeros(dim,length(labels));
model.Cov = zeros(dim,dim,length(labels));
for i=1:length(labels)
   inx = find(data.y==labels(i));
   n = length(inx);
   model.Mean(:,i) = sum(data.X(:,inx),2)/n;
   dataC = data.X(:,inx)-model.Mean(:,i)*ones(1,n);
   switch covType,
     case 'full', 
       model.Cov(:,:,i) = dataC*dataC'/n;
     case 'diag', 
       model.Cov(:,:,i) = diag(sum(dataC.^2,2)/n);
     case 'spherical'
       model.Cov(:,:,i) = eye(dim,dim)*sum(sum(dataC.^2))/(n*dim);
     otherwise
       error('Wrong covType.');
   end   
   model.Prior(i) = n/numData;
   model.y(i) = labels(i);
end
model.covType = covType;